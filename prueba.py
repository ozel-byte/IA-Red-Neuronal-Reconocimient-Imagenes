from PIL import Image
import numpy as np

class Prueba:
    listaImagenes = ["dataset/test/croton/codiaeum variegatum_01.png","dataset/test/chula/teresita_1.png","dataset/test/hojaLimon/limon_01.png","dataset/test/ixora/Ixora-coccinea_01.png","dataset/test/mango/Mango_01.png"]

    def __init__(self,model,categorias) -> None:
        self.inicarPrueba(model,categorias)

    def inicarPrueba(self,model,categorias):
        for x in self.listaImagenes:
            im = 0
            im = Image.open(x).resize((100,100))
            im = im.convert("RGB")
            im = np.asarray(im)
            im = np.array([im])

            predic = model.predict(im)
            print(categorias[np.argmax(predic)])
        pass