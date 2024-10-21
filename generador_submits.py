#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:45:14 2024

@author: walter
"""

import os

# Función para leer un archivo, buscar y reemplazar una línea, y generar un nuevo archivo
def reemplazar_linea_y_generar_nuevo_archivo(ruta_archivo_original, texto_reemplazo, ruta_archivo_nuevo):
    # Leer todas las líneas del archivo original
    with open(ruta_archivo_original, 'r') as archivo:
        lineas = archivo.readlines()

    # Abrir o crear un nuevo archivo en modo escritura
    with open(ruta_archivo_nuevo, 'w') as archivo_nuevo:
        for linea in lineas:
            # Si la línea contiene el texto que buscamos, la reemplazamos
            if linea.split('"')[0] in list(texto_reemplazo.keys()):
                archivo_nuevo.write(texto_reemplazo[linea.split('"')[0]] )
            else:
                archivo_nuevo.write(linea)

# Uso del código
pulso = 50
rangos =[7,8,9]
a = 50
betas = [
    1,
    5,
    10,
    50,
    100,
    500,
    1000,
    5000,
    10000,
    50000,
    100000,
    500000,
    1000000,
    5000000
    ]

ruta_archivo_original = '/home/walter/Documents/paper_crossbars/submits/submit1.sh'  # Cambia esto por la ruta de tu archivo original

for rango in rangos:
    nueva_carpeta = f'/home/walter/Documents/paper_crossbars/submits/no_lineal_pulso_{pulso}_rango_{rango}_a_{a}'
    os.mkdir(nueva_carpeta)

    for beta in  betas: 
        texto_reemplazo  = { 'pulso=' : f'pulso="{pulso}"\n' , 'rango=' : f'rango="{rango}"\n' , 'a=' : f'a="{a}"\n' , 'beta=' : f'beta="{beta}"\n' }
        ruta_archivo_nuevo = nueva_carpeta + f'/submit_{beta}.sh'  # Ruta del archivo nuevo que se creará
        # Llamar a la función para generar el nuevo archivo con la línea modificada
        reemplazar_linea_y_generar_nuevo_archivo(ruta_archivo_original, texto_reemplazo, ruta_archivo_nuevo)
        
        
        