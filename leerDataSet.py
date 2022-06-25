import os
from PIL import Image
import numpy as np
import tensorflow as tf


def cargarDataSet(carpeta):
    x = 0
    imagenes = []
    labels   = []
    categoria = os.listdir("dataset/"+carpeta)
    for nombreCarpeta in categoria:
        for imagen in os.listdir("dataset/"+carpeta+"/"+nombreCarpeta):
            img = Image.open("dataset/"+carpeta+"/"+nombreCarpeta+"/"+imagen).resize((100,100))
            img = img.convert("RGB")
            img = np.asarray(img)
            imagenes.append(img)
            labels.append(x)
        x+=1 
    return (imagenes,labels,categoria)