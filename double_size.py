# Ce programme double la taille d'une image en assayant de ne pas trop pixeliser l'image.
from PIL import Image
import os
import numpy as np
from scipy import signal
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

def process_block(block):
    maskG = np.array([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]]) / 16.
    blur = np.zeros_like(block, dtype=np.double)
    for i in range(3):
        blur[:,:,i] = signal.convolve2d(block[:,:,i], maskG, mode='same')
    maskS = np.array([[0, -1, 0],[-1,5,-1],[0,-1,0]])
    sharp = np.zeros_like(block, dtype=np.double)
    sharp[:,:,:2] = blur[:,:,:2]
    sharp[:,:,2] = np.clip(signal.convolve2d(blur[:,:,2], maskS, mode='same'), 0., 1.)
    return sharp

if rank == 0:
    start = MPI.Wtime()
    path = "datas/"
    image_path = path + "paysage2.jpg"
    img = Image.open(image_path)
    # Réduire l'image de moitié pour économiser de la mémoire
    new_width = img.width // 2
    new_height = img.height // 2
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS) 
    img = img.convert('HSV')
    img = np.array(img, dtype=np.double)
    img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1) / 255.
    H, W, _ = img.shape

    block_list = []
    for i in range(nbp):
        core_start = i * (H // nbp)
        core_end = H if i == nbp - 1 else (i + 1) * (H // nbp)
        core_rows = core_end - core_start
        if i == 0:
            if nbp == 1:
                block = img[0:core_end]
                offset_top = 0
                offset_bottom = block.shape[0]
            else:
                block = img[0:core_end + 1]
                offset_top = 0
                offset_bottom = core_rows
        elif i == nbp - 1:
            block = img[core_start - 1:H]
            offset_top = 1
            offset_bottom = 1 + core_rows
        else:
            block = img[core_start - 1:core_end + 1]
            offset_top = 1
            offset_bottom = 1 + core_rows
        block_list.append((block, offset_top, offset_bottom))
else:
    block_list = None

local_data = comm.scatter(block_list, root=0)
local_block, offset_top, offset_bottom = local_data
local_result = process_block(local_block)
local_core = local_result[offset_top:offset_bottom, :, :]
gathered = comm.gather(local_core, root=0)

if rank == 0:
    final_array = np.vstack(gathered)
    final_array = (255. * final_array).astype(np.uint8)
    final_img = Image.fromarray(final_array, 'HSV').convert('RGB')
    end = MPI.Wtime()
    print("Temps :", end - start)
    if not os.path.exists("sorties"):
        os.makedirs("sorties")
    final_img.save("sorties/paysage_double.jpg")
    print("Image sauvegardée")
