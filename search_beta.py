#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:55:03 2024

@author: walter
"""



import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
# from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import numpy as np
from func_crossbars import *
import matplotlib.pyplot as plt
import os
import time  

#Ti beta 1.6e6
#limita la cantidad de cores que usa pytorch
#torch.set_num_threads(6)

"""
    -Entrenamiento de crossbars con k-folds y batchs training implementado.
    -Con device = torch.device("cpu") se puede cambiar el device en el que corre (CPU o GPU).
    -RECORDAR!!!! si uso crossentropy no hay que poner explicitamente softmax. 
    -Haciendo entrenamiento en batchs la funcion de perdida se calcula como el promedio entre 
    los batchs. 

"""

'Defino los parámetros'

sistema = 'lineal'
pulsos = 100
rango =8
pot = np.load(f'/home/walter/Documents/Doctorado/Simulaciones Crossbars/curvas_sinteticas/pot_dep_lineales/pot_lineal_{pulsos}_pulsos_{rango}.npy')
dep = np.load(f'/home/walter/Documents/Doctorado/Simulaciones Crossbars/curvas_sinteticas/pot_dep_lineales/dep_lineal_{pulsos}_pulsos_{rango}.npy')
betas =    [ 1e5,5e4, 3e4, 2.3e4, 2e4, 1.8e4, 1.5e4, 1e3]
G0_distrbtn = 'random'
realizaciones = 1
epochs = 1
save_data  = False
k_folds = 5
batch_number = 32
porcentaje_maximo= 0
porcentaje_minimo = 0
D_in,   D_out = 784, 10
lr = 1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

pot = torch.from_numpy(pot)
dep = torch.from_numpy(dep)
pot, dep = pot.to(device), dep.to(device)


'Cargo los datos del MNIST y armo el objeto dataset de pytorch'
X_train = np.load('X_train_mnist.npy')
X_test = np.load('X_test_mnist.npy')
y_train = np.load('y_train_mnist.npy')
y_test = np.load('y_test_mnist.npy')
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Assuming y_train holds integer class labels
# Create a TensorDataset

#data y labels originales al GPU
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
mnist_dataset = TensorDataset(X_train_tensor, y_train_tensor)


'Traigo funciones de pytorch'
loss_cross_entropy = torch.nn.CrossEntropyLoss()

# 4. Set up K-Fold Cross-Validation
images_per_fold = len(X_train)/k_folds
batch_size = int((len(X_train) - images_per_fold)/batch_number)
batch_size_val = int(images_per_fold/5)
# KFold from sklearn
kf = KFold(n_splits=k_folds, shuffle=True)


for beta in betas : 
    acc_result = torch.zeros([k_folds*realizaciones, epochs])
    train_loss_result = torch.zeros([k_folds*realizaciones, epochs])
    val_loss_result = torch.zeros([k_folds*realizaciones, epochs])
    
    realizacion = 0
    
    for _ in range(realizaciones):
        for fold, (train_idx, val_idx) in enumerate(kf.split(mnist_dataset)):
            print(realizacion)
            print(f'Fold {fold+1}/{k_folds}')
            time_stamp = time.time()
            
        
            # Subset the dataset for the current fold
            train_subsampler = Subset(mnist_dataset, train_idx)
            val_subsampler = Subset(mnist_dataset, val_idx)
        
            #armo los dataloaders para aplicar batchs
            train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)
               
            G = G0_distribution(G0_distrbtn,pot, dep , D_in , D_out  , device = device)
            
            for e in range(1, epochs+1):
                # Hago el entrenamiento pasando los batchs
                l_batch = 0
                for data, target in train_loader:
                  
                    w = torch.zeros([D_in, D_out],requires_grad=False)
                    w = w.to(device)
                    for i in range(D_out):
                        w[:,i] = G[:,2*i] - G[:,2*i+1]    
                    
                
                            
                    w.requires_grad=True
                    
                    output = forward_MemSP(data,w,beta)
                    
                    # loss
                    loss = loss_cross_entropy(output, target)
                    l_batch +=  loss.item()
                    
                    # Backprop (calculamos todos los gradientes automáticamente)
                    loss.backward()
                    
                    
                    with torch.no_grad():
                        # update pesos
                        #w -= lr * w.grad
                        #
                        d_w = -lr * w.grad
                        G = Nmanhattan(d_w, G , pot,dep, D_in,D_out)
                        
                        # ponemos a cero los gradientes para la siguiente iteración
                        # (sino acumularíamos gradientes)
                        w.grad.zero_()
                #epoch complited              
                accuracy, val_loss = evaluate_folds(w, beta, device, val_loader)
                #meto los datos de acc, loss_train y loss_val para cada entrenamiento en una matriz 
                acc_result[realizacion,e-1] = accuracy
                train_loss_result[realizacion,e-1] = l_batch/batch_number
                val_loss_result [realizacion,e-1] = val_loss
                print(f"Epoch {e}/{epochs} --- Loss train {l_batch/batch_number:.5f} --- Loss val {val_loss:.5f} --- Acc {accuracy:.3f}")
                realizacion += 1
    
    mean_curve =acc_result.mean(dim= 0)
    
    # Convertir el tensor a NumPy para usar con Matplotlib
    tensor_np = output.detach().numpy()
    
    # Crear un histograma con Matplotlib
    plt.hist(tensor_np.flatten(), bins=20, edgecolor='k', alpha=0.7,label = f'beta = {beta}, mean acc = {mean_curve.item():1.1f}, hist mean = {tensor_np.flatten().mean():1.1f},hist std = {tensor_np.flatten().std():1.1f} ')
    plt.title("Histograma de Tensor")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.show()       