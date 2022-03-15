# Bibliotecas para utilização

import numpy as np # Trabalhando com Matrizes
import cv2 # Para visão computacional
import time #????
from os.path import splitext, basename, join #?????

# Criando as Classes Necessárias 

class Colorizador: # Função para colorir as Imagens de Acordo com o Modelo Pré-Setado

    def __init__(self, height = 480 , width = 600): # Parametros usados durantes as funções
        (self.height, self.width) = height, width

        self.colorModel = cv2.dnn.readNetFromCaffe("modelos/colorization_deploy_v2.prototxt" , caffeModel = "modelos/colorization_release_v2.caffemodel") # Leitura dos modelos pré carregados

        clusterCenters = np.load("modelos/pts_in_hull.npy")
        clusterCenters = clusterCenters.transpose().reshape(2, 313, 1, 1)

        self.colorModel.getLayer(self.colorModel.getLayerId('class8_ab')).blobs = [clusterCenters.astype(np.float32)]
        self.colorModel.getLayer(self.colorModel.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313] , 2.606, dtype="float32")]


    def processaImagem(self, imgName): # Função para Leitura e Dimensionamento das Imagens

        self.img = cv2.imread(imgName)
        self.img = cv2.resize(self.img, (self.width , self.height))


        self.processaFrame()

        cv2.imwrite(join("saida_colorida" , basename(imgName)) , self.imgFinal)

        cv2.imshow("saida_colorida" , self.imgFinal)



    def processaFrame(self): # Função para Processar os Frames

        imgNormalized = (self.img[:,:,[2,1,0]] * 1.0 /255).astype(np.float32) # Normalizando a Imagem

        imgLab = cv2.cvtColor(imgNormalized, cv2.COLOR_RGB2Lab) # Convertendo RGB images para Color Lab Spaces -> Ver Refência sobre CIELAB

        channelL = imgLab[:,:,0] # Contêm as informações de iluminação de acordo com CIELAB


        imgLabResized = cv2.cvtColor(cv2.resize(imgNormalized , (224 , 224)) , cv2.COLOR_RGB2Lab) # Modificar Tamanho da Imagem Normalizada

        channelLResized = imgLabResized[:,:,0]

        channelLResized -= 50

        # Imagem agora está pronta para ser convertida de gray scale para colorfull

        self.colorModel.setInput(cv2.dnn.blobFromImage(channelLResized))

        result = self.colorModel.forward()[0,:,:,:].transpose((1,2,0))

        resultResized = cv2.resize(result, (self.width, self.height)) # Colocando na resolução original


        self.imgOut = np.concatenate((channelL[:,:,np.newaxis] , resultResized), axis = 2)

        self.imgOut = np.clip(cv2.cvtColor(self.imgOut, cv2.COLOR_LAB2BGR), 0 , 1) # Clipando os Valores entre 0 e 1

        self.imgOut = np.array((self.imgOut) * 255, dtype = np.uint8)


        self.imgFinal = np.hstack((self.img, self.imgOut)) # Comparação lado a lado

