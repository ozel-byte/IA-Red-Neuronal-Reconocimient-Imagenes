import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
    
def grafica2(train):
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train[0])
    plt.xlabel("limon")
    plt.show()
            

def crearModelo2(img,labels):
    modelo = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation='sigmoid', input_shape=(100, 100,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64,(3,3),activation='sigmoid'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(20, activation="softmax")

    ])

    modelo.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
   
    img = np.asanyarray(img,dtype="float32")
    labels = np.asarray(labels,dtype="float32")
    
    history=modelo.fit(
        img,
        labels,
        epochs=40,
        batch_size=32
    )
    plt.plot(history.history['loss'])
    plt.show()
    
    pass