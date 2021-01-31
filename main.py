#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#####################################################
#                                                   #
#           INDEXACIÓN Y RECUPERACIÓN DE            #
#                    IMÁGENES                       #
#           Mª del Mar Alguacil Camarero            #
#                                                   #
#####################################################

import cv2
import unittest
import numpy as np
from matplotlib import pyplot as plt
from math import exp, ceil, floor, sqrt
from funciones import *
from auxFunc import *


############################
# EJERCICIO 1
############################

# EMPAREJAMIENTO DE DESCRIPTORES
# Leer parejas de imágenes que tengan partes de escena comunes. 
# Haciendo uso de una máscara binaria o de las funciones extractRegion() y 
# clickAndDraw(), seleccionar una región en la primera imagen que esté presente 
# en la segunda imagen. Para ello solo hay que fijar los vértices de un polígono que contenga a la región. 
# Extraiga los puntosSIFT contenidos en la región seleccionada de la primera 
# imagen y calcule las correspondencias con todos los puntos SIFT de la segunda 
# imagen. Pinte las correspondencias encontrados sobre las imágenes. Jugar con 
# distintas parejas de imágenes y decir que conclusiones se extraen de los 
# resultados obtenidos con respecto a la utilidad de esta aproximación
# en la recuperación de imágenes a través de descriptores.
  
def emparejamiento(filenames, leer=True):
    # Leemos y pasamos las imágenes a blanco y negro
    if leer:
        print(filenames[0])
        img1 = cv2.imread(filenames[0])
        img2 = cv2.imread(filenames[1])
    else:
        img1 = filenames[0]
        img2 = filenames[1]

    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    # Seleccionamos las regiones que queremos tener en cuenta
    region1 = extractRegion(gray1)
    region2 = extractRegion(gray2)
    
    # Calculamos las máscaras que deben ser del mismo tamaño que las respectivas
    # imágenes. 
    img_nula1 = np.zeros(gray1.shape)
    mask1 = cv2.fillConvexPoly(img_nula1, np.asarray(region1), 255)
    
    img_nula2 = np.zeros(gray2.shape)
    mask2 = cv2.fillConvexPoly(img_nula2, np.array(region2), 255)

    # Calculamos los keypoints SIFT con dicha máscara y dibujamos las distintas 
    # imágenes con las correspondencias obtenidas.
    emparejarSift([img1, img2], dibujar=True, leer=False, N=100, M=-1, f=0.65,
                    mask1=np.array(mask1, dtype=np.uint8), 
                    mask2=np.array(mask2, dtype=np.uint8), ej1=False)


############################
# EJERCICIO 2
############################
#VISUALIZACIÓN DEL VOCABULARIO
# Usando las imágenes dadas se han extraido regiones de cada imagen y se ha 
# construido un vocabulario de palabras usando k-means. Se han extraído de 
# forma directa las regiones imágenes asociadas y se han re-escalado. Elegir al 
# menos dos palabras visuales diferentes y visualizar las regiones imagen de los 
# N parches más cercanos de cada palabra visual, de forma que se muestre el 
# contenido visual que codifican.
def visVocabulario(vocabulario, regiones, p1, p2, N=15):
    # Leemos los ficheros
    descriptores, patches = loadAux(regiones, True)
    precision, etiquetas, palabras = loadDictionary(vocabulario)
    
    # Pasamos los parches a blanco y negro
    parches = []
    for patch in patches:
        parches.append(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY))
    
    # Matriz con las etiquetas(cluster al que pertenece) y la posición de los descriptores y los parches
    tam = len(parches)
    labels = np.squeeze(np.asarray((etiquetas)))[0:tam]
    labels = np.array([labels, list(range(tam))]).T

    labelsP1 = labels[np.where(labels[:,0] == p1)][:,1]
    labelsP2 = labels[np.where(labels[:,0] == p2)][:,1]

    # Normalizamos los descriptores para que tengan norma cuadrática igual a uno
    norma = np.sqrt((descriptores**2).sum(axis=1))
    norma = np.matrix([norma]*descriptores.shape[1]).T

    descriptores = np.array(np.divide(descriptores, norma))

    # Vector distancias
    distanciasP1 = np.matrix(palabras[p1]) * descriptores[labelsP1].T
    distanciasP1 = np.squeeze(np.asarray(distanciasP1))
    distanciasP2 = np.matrix(palabras[p2]) * descriptores[labelsP2].T
    distanciasP2 = np.squeeze(np.asarray(distanciasP2))

    # Matriz que contiene las distancias calculadas y la posición asociada
    distPosP1 = np.array([distanciasP1, labelsP1]).T  
    distPosP2 = np.array([distanciasP2, labelsP2]).T 
    
    # Ordenamos las distancias
    distPosP1 = distPosP1[distPosP1[:, 0].argsort()]
    distPosP1 = distPosP1[::-1]
    distPosP2 = distPosP2[distPosP2[:, 0].argsort()]
    distPosP2 = distPosP2[::-1]
    
    # Extraemos los índices de los N parches más cercanos
    S = min(N, distPosP1.shape[0])
    indices1 = np.array(distPosP1[0:S,1], dtype = np.int32)
    T = min(N, distPosP2.shape[0])
    indices2 = np.array(distPosP2[0:T,1], dtype = np.int32)
    
    # Convertimos la lista de parches en un array
    parches = np.array(parches)
    
    # Visualizar las regiones imagen de los N parches más cercanos de cada palabra visual
    visualization(parches[indices1], ['']*S, 3, S/3, color = False)
    visualization(parches[indices2], ['']*T, 3, T/3, color = False)
    

# Varianza de la palabra deseada
def varianza(descriptores, palabra):
    varianzas = np.squeeze(np.asarray(palabra*descriptores.T))
        
    return max(varianzas)

# Seleccionamos las num_palabras con menor dispersión
def varianzas(vocabulario, regiones, num_palabras=2):
    # Leemos los ficheros
    descriptores, parches = loadAux(regiones, True)
    precision, etiquetas, palabras = loadDictionary(vocabulario)
    
    # Normalizamos los descriptores para que tengan norma cuadrática igual a uno
    norma = np.sqrt((descriptores**2).sum(axis=1))
    norma = np.matrix([norma]*descriptores.shape[1]).T

    descriptores = np.array(np.divide(descriptores, norma))
    
    # Matriz con las etiquetas, los descriptores y los parches
    tam = len(parches)
    labels = np.squeeze(np.asarray((etiquetas)))[0:tam]
    labels = np.array([labels, list(range(tam))]).T

    # Varianzas de los distintos clusters
    N = palabras.shape[0]
    var = [None]*N
    for p in range(N):
        labelsP = labels[np.where(labels[:,0] == p)][:,1]
        var[p] = varianza(descriptores[labelsP], np.matrix(palabras[p]))
    var = np.array([var, list(range(N))]).T
    var = var[var[:, 0].argsort()]
    var = var[::-1]
    
    # Devolvemos los índices de las palabras
    return np.array(var[:num_palabras, 1], dtype=int)
    
    
############################
# EJERCICIO 3
############################ 

# RECUPERACIÓN DE IMÁGENES
# Implementar un modelo de índice invertido + bolsa de palabras para las imágenes 
# dadas en imagenesIR.rar usando el vocabulario.
# Verificar que el modelo construido para cada imagen permite recuperar imágenes
# de la misma escena cuando la comparamos al resto de imágenes de la base de datos.
# Elegir dos imágenes-pregunta en las que se ponga de manifiesto que el modelo
# usado es realmente muy efetivo para extraer sus semejantes y elegir otra 
# imagen-pregunta en la que se muestre que el modelo puede realmente fallar. 
# Para ello muestre las cinco imágenes más semejantes de cada una de las 
# imágenes-pregunta seleccionadas usando como medida de distancia el producto 
# escalar normalizado de sus vectores de bolsa de palabras.

# Para todas las imagenes les saco los descriptores, bolsa de palabras para cada una
# fichero invertido: 
def ficheroInvertido(filenames, vocabulario, umbral = 1.7):
    # Leemos todas la imagenes 
    imagenes = []
    for file in filenames:
        imagenes.append(cv2.imread(file, 0))
    
    # Leemos el vocabulario
    precision, etiquetas, palabras = loadDictionary(vocabulario)
    
    # Inicializamos SIFT
    sift = cv2.xfeatures2d.SIFT_create()   
        
    # Detectamos los keypoint y extraemos los descriptores de las distintas imágenes
    desc = []
    for img in imagenes:
        kpts, dcts = sift.detectAndCompute(img, None)
        desc.append(dcts)


    # Creamos una tabla de indice invertido de tamaño igual al número de palabras
#    fichero = np.empty(palabras.shape[0], dtype='O') # 'O' -> (Python) objects
    fichero = [np.copy([])]*palabras.shape[0]    
    for j, descriptor in enumerate(desc):
        # Normalizamos los descriptores
        max_desc = np.max(descriptor)
        descriptor = np.matrix(descriptor/max_desc)

        # Matriz de distancias
        distancia = palabras*descriptor.T 

        # Seleccionamos la distancia mayor
        # Máximo de cada fila (posición)
        maximos = np.argmax(distancia, axis=0)
        
        for i in range(maximos.shape[1]):
            maximo = maximos[0,i]
            if distancia[maximo,i] > umbral:
                fichero[maximo] = np.append(fichero[maximo], j)

    # BOLSA DE PALABRAS
    N = len(imagenes)
    # Incializamos los histogramas con 0's
    histogramas = np.zeros((N, len(fichero)))
    for j, palabra in enumerate(fichero):
        if len(palabra)!=0: # Si existe alguna imagen con esa palabra
            palabra = np.array(palabra)
            for i in range(N):
                histogramas[i,j] = len(np.where(palabra==i)[0])
            
    # Normalizamos el histograma
    for i in range(N):
        maxi = max(histogramas[i])
        histogramas[i] /= maxi
    

    return histogramas
    
   
def recuperacion(filenames, vocabulario, indice, N=5):
    histogramas = ficheroInvertido(filenames, vocabulario)
    # Comparamos histogramas y elegimos las N imagenes mas parecidas
    errores = []
    h = histogramas[indice]
    for i,g in enumerate(histogramas):
        if i!=indice:
            e = np.dot(g, h)/(sqrt(np.dot(g, g))*sqrt(np.dot(h, h)) )
            errores.append((e,i))
            
    errores.sort(key=lambda x: x[0], reverse=True)
    
    mejores = list(map(lambda x: x[1], errores[0:N]))
    
    hist = [histogramas[indice]]
    img = [cv2.imread(filenames[indice])]
    for i in range(N):
        ind = mejores[i]
        img.append( cv2.imread(filenames[ind]) )
        hist.append( histogramas[ind] )
        
    # Visualizamos las distintas imagenes y los histogramas asociados    
    visualization(img, ['']*(N+1), 2, (N+1)/2, color = True)
    visualization(hist, ['']*(N+1), 2, (N+1)/2, plot=True)
    
class TestPractica3(unittest.TestCase):
    def setUp(self):
        pass

     # EJERCICIO 1
    def test_emparejamiento(self):
        filename = ['./imagenes/36.png','./imagenes/50.png']
        emparejamiento(filename)
        
     # EJERCICIO 2
    def test_visVocabulario(self):
        vocabulario = 'kmeanscenters5000.pkl'
        regiones = 'descriptorsAndpatches.pkl'
        
        palabras = varianzas(vocabulario, regiones)
        #print(palabras)
        visVocabulario(vocabulario, regiones, palabras[0], palabras[1] )
        
        palabra1 = 1662
        palabra2 = 500
        visVocabulario(vocabulario, regiones, palabra1, palabra2 )
     
    # EJERCICIO 3
    def test_recuperacion(self):
        vocabulario = 'kmeanscenters5000.pkl'
        filenames = []
        for i in range(440):
            filenames.append('./imagenes/'+str(i)+'.png')
        
        recuperacion(filenames, vocabulario, 2)
      
if __name__ == '__main__':
    unittest.main()
