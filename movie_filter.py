# Ce programme va charger n images et y appliquer un filtre de netteté
# puis les sauvegarder dans un dossier de sortie
from PIL import Image
import os
import numpy as np
from scipy import signal

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

# Fonction pour appliquer un filtre de netteté à une image
def apply_filter(image):
    # On charge l'image
    img = Image.open(image)
    #print(f"Taille originale {img.size}")
    # Conversion en HSV :
    img = img.convert('HSV')
    # On convertit l'image en tableau numpy et on normalise
    img = np.repeat(np.repeat(np.array(img), 2, axis=0), 2, axis=1)
    img = np.array(img, dtype=np.double)/255.
    #print(f"Nouvelle taille : {img.shape}")
    # Tout d'abord, on crée un masque de flou gaussien
    mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    # On applique le filtre de flou
    blur_image = np.zeros_like(img, dtype=np.double)
    for i in range(3):
        blur_image[:,:,i] = signal.convolve2d(img[:,:,i], mask, mode='same')
    # On crée un masque de netteté
    mask = np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]])
    # On applique le filtre de netteté
    sharpen_image = np.zeros_like(img)
    sharpen_image[:,:,:2] = blur_image[:,:,:2]
    sharpen_image[:,:,2] = np.clip(signal.convolve2d(blur_image[:,:,2], mask, mode='same'), 0., 1.)

    sharpen_image *= 255.
    sharpen_image = sharpen_image.astype(np.uint8)
    # On retourne l'image modifiée
    return Image.fromarray(sharpen_image, 'HSV').convert('RGB')

# On crée un dossier de sortie
if not os.path.exists("sorties/perroquets"):
    os.makedirs("sorties/perroquets")
out_path = "sorties/perroquets/"

# Le processus 0 prépare les lots d'images
if rank == 0:
    start_time = MPI.Wtime()
    path = "datas/perroquets"
    N = 37 

    image_list = [(os.path.join(path, "Perroquet{:04d}.jpg".format(i+1)), i+1) for i in range(N)]  # (chemin_image, index)
    
    blocs = []
    bloc_size = len(image_list) // nbp
    r = len(image_list) % nbp
    start = 0
    for i in range(nbp):               # répartition équitable des images entre les processus
        extra = 1 if i < r else 0
        end = start + bloc_size + extra
        blocs.append(image_list[start:end])
        start = end
else:
    blocs = None

local = comm.scatter(blocs, root=0)    # envoi du lot d'images à chaque processus

for image_path, id in local:
    result_img = apply_filter(image_path)
    output_filename = os.path.join(out_path, "Perroquet{:04d}.jpg".format(id))
    result_img.save(output_filename)

comm.Barrier()  # on synchronise pour s'assurer que tous les processus ont terminé leur calcul


if rank == 0:
    end_time = MPI.Wtime()
    print(f"Temps d'exécution total : {end_time - start_time:.2f} secondes.")
