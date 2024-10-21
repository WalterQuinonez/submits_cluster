#!/bin/bash

# Script para correr trabajo serial

# Directorio actual es el raiz
#$ -cwd

# Nombre del proceso
#$ -N Crossbars_lineal_30pulsos_rango_7_a_10

# stdout y stderr al mismo archivo de salida
#$ -j y

# Usar bash como shell para los comandos que se ejecutaran
#$ -S /bin/bash

# Pido la cola a usar
#$ -q caulle
#$ -q copahue
# Pido 1GB RAM para el proceso (obligatorio)
#$ -l mem=1.5G
#$ -pe smp 6

# Las variables de entorno actual son incluidas en el trabajo
#$ -V

# dejo el nombre del directorio actual en la variable DirActual
DirActual=$(pwd)
pulso="30"
rango="7"
a="100"
beta="1"
epochs="1000"
realizaciones="50"

#exit
# Paso archivos al directorio temporal $TMP

cp $DirActual/DNN_experimental_NManhattan_rango_var.py /$TMP
cp $DirActual/func_crossbars.py /$TMP
cp $DirActual/compilado.sh /$TMP


# Entro al directorio temporal
cd $TMP

#CorroLoQueMeInteresa
conda activate Pytorch
bash compilado.sh $pulso $rango $beta $epochs $realizaciones $a


# Nombre del directorio de salida
DirSalida="lineal_pulso_${pulso}_rango_${rango}_a_${a}_beta_${beta}_epochs_${epochs}_realiz_${realizaciones}"
echo "Directorio de salida = $DirSalida"
# creo directorio de salida
mkdir "$DirActual/$DirSalida"
# muevo los archivos de salida al directorio de salida
mv * "$DirActual/$DirSalida/."
# muevo los .e y .o al directorio de salida
cd $DirActual
mv ${SGE_STDOUT_PATH} "$DirActual/$DirSalida/."
mv ${SGE_STDERR_PATH} "$DirActual/$DirSalida/."

rm "$DirActual/$DirSalida"/DNN_experimental_NManhattan_rango_var.py
rm "$DirActual/$DirSalida"/func_crossbars.py
rm "$DirActual/$DirSalida"/compilado.sh


# mv CrossEntropy_Softmax_accuracy_sintetica_lineal_beta_1.0.npy /$DirSalida
# mv CrossEntropy_Softmax_loss_sintetica_lineal_beta_1.0.npy /$DirSalida
