



from RedNeuronal import crearModelo2, grafica2
from leerDataSet import cargarDataSet


if __name__ == "__main__":
    imagenes,labels,categoria = cargarDataSet("train")
    crearModelo2(imagenes,labels)
    pass