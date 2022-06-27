import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from leerDataSet import cargarDataSet

from prueba import Prueba
import seaborn as sns
    

            

def crearModelo2(img,labels,categoria):
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation='sigmoid', input_shape=(100, 100,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64,(3,3),activation='sigmoid'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid'),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(10, activation="softmax")

    ])

    modelo.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
   
    img = np.asanyarray(img,dtype="float32")
    labels = np.asarray(labels,dtype="float32")
    
    history=modelo.fit(
        img,
        labels,
        epochs=100,
        
    )
    Prueba(modelo,categoria)
    plt.plot(history.history['loss'],label="loss")
    plt.title("loss")
    plt.show()
    graficar(modelo,history)


    pass


def graficar(modelo,historial):
    names = ["chula","croton","limon","ixora","mango"]
    imagenes,labels,categoria = cargarDataSet("test")
    audio = np.array(imagenes)
    labels = np.array(labels)
    y1 = np.argmax(modelo.predict(audio), axis=1)
    y2 = labels
    cfmtx = tf.math.confusion_matrix(y1,labels)
    fig = plt.figure(figsize=(10,8))
    sns.heatmap(
        cfmtx,
        xticklabels= names,
        yticklabels= names,
        annot=True,
        cmap='icefire',
        fmt='g'
    )
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.show()
    fig = plt.figure(figsize=(12, 7))
    print("llego aqui grafica matriz")
    pass