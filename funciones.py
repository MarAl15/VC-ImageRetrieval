#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#####################################################
#                                                   #
#                     FUNCIONES                     #
#           Mª del Mar Alguacil Camarero            #
#                                                   #
#####################################################

import cv2
import unittest
import numpy as np
from matplotlib import pyplot as plt
from math import exp, ceil

# REPRESENTACIÓN DE LAS IMÁGENES
    # images: lista de las imágenes leídas
    # titles: nombres asociados a images
    # row: número de filas que queremos
    # col: número de columnas que deseamos
    # color: True si es una imagen a color, False en caso contrario
    # RESTRICCIÓN: debe haber igual número de imágenes que de títulos y row*col debe ser también igual a este.
def visualization(images, titles, row, col, color = False, plot=False):
    sizeI = len(images) # Tamaño de la lista de imágenes que queremos visualize
    
    # Comprobamos que los datos introducidos son correctos
    assert sizeI==row*col and sizeI==len(titles)
        
    for i in range(sizeI):
        plt.subplot(row, col, i+1)
        
        if not plot:
            if not color: # Imagen a escala de grises
                plt.imshow(images[i], "gray")
            else: # Imagen a color
                imagen = np.uint8(images[i])
                rgb_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                
                plt.imshow(rgb_img)
        else:
            plt.plot(images[i])
            
        
        # Ocultamos los valores de los ejes 
        plt.xticks([]), plt.yticks([])
        
        # Asignamos un título a la imagen
        plt.title(titles[i])
    
    # Mostramos la composición de imágenes resultante
    plt.show()
    
    # Punto de parada
    cv2.waitKey(0) 
   
    
#-----------------------------------------------------------------------------------------------------------
    # filename: nombre de la imagen
    # sigma: desviación típica de la Gaussiana
    # border_type: tipo de borde
    # size: tamaño del kernel gaussiano
    # color: True si la imagen queremos que sea procesada a color, y False en caso contrario
def convol_gaussiana(filename, sigma, border_type=cv2.BORDER_DEFAULT, size=None, color=False, read=True):
    # Lectura de la imagen
    if read:
        if color:
            image = cv2.imread(filename)
        else:
            image = cv2.imread(filename, 0)
    else:
        image = filename
        
    # Si no se ha especificado size o alguno de ellos no es impar le asignamos el tamaño óptimo
    if not size or (size[0]%2==0 or size[1]%2==0): 
        opt = int(6*sigma)+1 #3*sigma*2+1
        size = (opt, opt)
      
    # Alisamiento gaussiano
    gauss = cv2.GaussianBlur(src=image, ksize=size, sigmaX=sigma, borderType=border_type)
    
    # Reparametrizamos para que los colores estén comprendidos en el intervalo [0,255]            
    gauss = rangoColor(gauss, color)
    
    # Visualizamos las imágenes
    visualization([image, gauss], ['Original', 'Gaussiana'], 1, 2, color)
    
    return gauss
    
#-----------------------------------------------------------------------------------------------------------
    
#CONVOLUCION
    
# Cambiamos el intervalo de los colores
# [a,b] -> [0,255]
def cambio(x, a, b):
    return round(255*(x-a)/(b-a))

def rango(matrix):
    minimo = min(map(min,matrix))
    maximo = max(map(max,matrix))
    
#    for x in matrix:
#        x = cambio(x, minimo, maximo)
#        print(x)
    if minimo<0 or maximo>255:
        # [minimo, maximo] -> [0,255]
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i,j] = cambio(matrix[i,j], minimo, maximo)
    else: # Redondeamos sólo
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i,j] = int(round(matrix[i,j]))


def rangoColor(matrix, color=False):
    if not color:
        # Cambiamos el rango de la escala de grises
        rango(matrix)
    else:
        # Obtenemos las tres matrices de colores
        b,g,r = cv2.split(matrix)
        
        # Cambiamos el rango de cada color
        rango(b)
        rango(g)
        rango(r)
        
        # Mezclamos los colores
        matrix = cv2.merge([b,g,r])
    
    return matrix
        
    # image: archivo imagen
    # kernelX: vector fila de la máscara separable
    # kernelY: vector columna de la máscara separable
    #       máscara = kernelX*kernalY 
    # border_type: tipo de borde
    # change: True si se desea que el intervalo de definición de la imagen 
    #convolucionada sea [0,255], y False en caso contrario.
def convolucion(image, kernelX, kernelY, border_type=cv2.BORDER_DEFAULT, change=True):
    row = image.shape[0] # Número de filas de la imagen
    col = image.shape[1] # Número de columnas

    blur = 1.0*image.copy() # Copiamos la imagen

    # Convolución por filas 
    for i in range(row):
        v = cv2.filter2D(src=blur[i,:], ddepth=-1, kernel=kernelX, borderType=border_type)
        blur[i,:] = v[:,0]

    # Convolución por columnas
    for j in range(col):
        v = cv2.filter2D(src=blur[:,j], ddepth=-1, kernel=kernelY, borderType=border_type)
        blur[:,j] = v[:,0]
      
    # Cambiamos al intervalo [0,255]
    if(change):
        rangoColor(blur)
    
    # Devolvemos la matriz convolucionada
    return blur

    # filename: nombre de la imagen
    # border_type: tipo de borde
    # kernelX: vector fila de la máscara separable
    # kernelY: vector columna de la máscara separable
    #           máscara = kernelX*kernalY 
    # sigma: desviación típica gaussiana 
    # visualize: True si queremos que se visualice la imagen resultante, False en caso contrario
def convol_separable(filename, border_type=cv2.BORDER_DEFAULT, kernelX=[], kernelY=[], sigma=None, 
                     visualize=True, titles=['Original', 'Nucleo separable'], read=True):
    if read:
        image = cv2.imread(filename, 0)
    else:
        image = filename.copy()
    
    if sigma:
        opt = int(6*sigma)+1
        # cv2.getGaussianKernel -> Si sigma no es positiva sigma=0.3*(ksize-1)*0.5-1)+0.8)
        kernel = cv2.getGaussianKernel(ksize=opt, sigma=sigma)
        
        # Alisamiento gaussiano
        blur = convolucion(image, kernel, kernel, border_type, False)
    else:
        blur = image.copy()

#    if kernelX==None or kernelY==None:
#    if list(np.concatenate(kernelX)) and list(np.concatenate(kernelY)):
    if len(kernelX)!=0 and len(kernelY)!=0:
        # Aplicamos el filtro lineal separable a la imagen 
        blur = convolucion(blur, kernelX, kernelY, border_type)
    
    
    # Cambiamos al intervalo [0,255] para visualizarla
    blur = rangoColor(blur)
    
    # Visualizamos las imágenes resultantes
    if visualize:
        visualization([image, blur], titles, 1, 2)

    # Devolvemos la imagen convolucionada
    return blur

#-----------------------------------------------------------------------------------------------------------
 
# CONVOLUCION CON NUCLEO DE 1º DERIVADA
    # filename: nombre de la imagen
    # sigma: desviación típica de la gaussiana
    # border_type: tipo de borde        
def convol_1derivada(filename, sigma, border_type=cv2.BORDER_DEFAULT, visualize=True, read=True):
    # Tamaño de la máscara
    if sigma<=1:
        tamanio = int(6*sigma)+1
    else:
#        sigma = 1
        tamanio = 7
    
    # Obtenemos los vectores de la máscara
    kernelDX = cv2.getDerivKernels(1,0, tamanio)
    kernelDY = cv2.getDerivKernels(0,1, tamanio)

    # Alisamiento gaussiano para intentar eliminar el ruido    
    # y convolución con núcleo de 1ª derivada:
    #   - Respecto de X
    blurX = convol_separable(filename, border_type, kernelDX[0], kernelDX[1], sigma, False, read=read)
        
    #   - Respecto de Y
    blurY = convol_separable(filename, border_type, kernelDY[0], kernelDY[1], sigma, False, read=read)
        
    # Visualizamos
    if visualize:
#        visualization(images, titles, row, col, color = False)
        
#        visualization([cv2.imread(filename, 0), blurX, blurY], 
#                      ['Original', 'Derivada resp. X', 'Derivada resp. Y'], 1, 3, color=False)
        visualization([ blurX, blurY], ['', ''], 1, 2, color=False)
        
    return blurX, blurY
    
#-----------------------------------------------------------------------------------------------------------
    
    
# PIRÁMIDE GAUSSIANA
    # pyr: Vector con las distintas imágenes que se quiere visualizar en forma de pirámide
def visualizationPyr(pyr):
    levels = len(pyr)
    
    rows = pyr[0].shape[0]
    cols = pyr[0].shape[1]
    
    # Añadimos columnas de ceros si el número de filas/columnas no es par
    if rows%2!=0:
        pyr[0] = np.concatenate( (pyr[0], np.zeros((1, cols))), axis=0)
        rows+=1
    
    if cols%2!=0:
        pyr[0] = np.concatenate( (pyr[0], np.zeros((rows, 1))), axis=1)
        cols+=1
    
    ipyr = 255*np.ones((rows, ceil(3*cols/2))) # Matriz de 255's -> Fondo blanco
    
    # Colocamos la imagen primera a la izquierda
    ipyr[:rows,:cols] = pyr[0]
        
    # El resto de las imágenes a la derecha desde arriba hacia abajo
    ri = 0 # Fila inicial
    for i in range(1,levels):         
        potencia  = pow(2,i)
        rf = ri+pyr[i].shape[0] #ceil( (potencia-1)*rows/potencia ) # Fila final
        cf = ceil( (potencia+1)*cols/potencia ) # Columna final -> cols+pyr[i].shape[1]#
            
        # Colocamos la nueva imagen en la posición correspondiente
        ipyr[ri:rf, cols:cf] = pyr[i]
                
        ri=rf # Fila inicial para la siguiente imagen a colocar

    # Visualizamos la pirámide construida
    visualization([ipyr], [''], 1, 1)
    
    
    # filename: nombre de la imagen
    # sigma: desviación típica de la gaussiana
    # border_type: tipo de borde
    # visualize: True si queremos que se visualice la imagen resultante, False en caso contrario
    # levels: número de niveles de la pirámide Gaussiana
    # read: True si la imagen debe ser leída y False en caso contrario
def pyrGauss(filename, border_type=cv2.BORDER_DEFAULT, visualize = True, levels=4, read=True):
    # Imagen
    if read:
        image = cv2.imread(filename,0)
    else:
        image = filename
        
    # Pirámide Gaussiana
    pyrGaussian = [image.copy()]
    for i in range(1, levels+1):
        pyrGaussian.append(cv2.pyrDown(src = pyrGaussian[i - 1], borderType=border_type))
    
    # Visualizamos
    if visualize:
        visualizationPyr(pyrGaussian)
        
    # Devolvemos la pirámide
    return pyrGaussian

########################################################################################################
#   Usar el detector-descriptor SIFT de OpenCV sobre las imágenes. Extraer sus 
# listas de keyPoints y descriptores asociados. Establecer las correspondencias existentes entre ellos 
# usando el objeto BFMatcher de OpenCV. Valorar la calidad de los resultados obtenidos en términos  
# de correspondencias válidas usando los criterios de correspondencia "BruteForce+crossCheck" y "Lowe-Average-2NN"
def emparejarSift(filenames, dibujar=True, leer=True, N=100, M=-1, f=0.65, mask1=None, mask2=None, ej1=False):  
    # Leemos y pasamos la imagenes a blanco y negro
    if leer:
        img1 = cv2.imread(filenames[0])
        img2 = cv2.imread(filenames[1])
    else:
        img1 = filenames[0]
        img2 = filenames[1]
        
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # Inicializamos SIFT
    sift = cv2.xfeatures2d.SIFT_create()   
        
    # Detectamos los keypoint y extraemos los descriptores de la foto
    kpts1, dcts1 = sift.detectAndCompute(gray1, mask1)
    kpts2, dcts2 = sift.detectAndCompute(gray2, mask2)
        
    # BFMatcher (int normType=NORM_L2, bool crossCheck=false)
    #BruteForce+crossCheck
    #crossCheck is an alternative to the ratio test. It breaks knnMatch. So either use crossCheck=False and then do 
    #the ratio test, or use crossCheck=True and use bf.match() instead of bf.knnMatch().
    bf = cv2.BFMatcher(crossCheck=True)
    matches1 = bf.match(dcts1,dcts2) # Descriptores de emparejamiento
    
    # Los ordenamos segun su distancia
    matches1 = sorted(matches1, key = lambda x:x.distance)
    
    if dibujar:
        if N>=len(matches1) or N<0:
            N=len(matches1)-1
            
        img_match = cv2.drawMatches(img1,kpts1, img2,kpts2, matches1[:N], None, flags=2)
        visualization([img_match], [''], 1, 1, color = True)
    
    # Lowe-Average-2NN
    bf = cv2.BFMatcher()
    matches2 = bf.knnMatch(dcts1,dcts2, k=2)
    
    aptos = []
    matches = []
    for m,n in matches2:
        if m.distance < f*n.distance:
            aptos.append([m])
            matches.append(m)
            
    matches2 = aptos

    if dibujar:
        if M>=len(matches2) or M<0:
            M=len(matches2)-1
            
        img_knnmatch = cv2.drawMatchesKnn(img1,kpts1, img2,kpts2, matches2[:M], None)
        visualization([img_knnmatch], [''], 1, 1, color = True)

    # Devolvemos los keyPoints y las correspondencias con Lowe-Average-2NN
    return kpts1, kpts2, matches