import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops

# https://youtu.be/5x-CIHRmMNY
"""
@author: Sreenivas Bhattiprolu

skimage.feature.greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
distances - List of pixel pair distance offsets.
angles - List of pixel pair angles in radians.

skimage.feature.graycoprops(P, prop)
prop: The property of the GLCM to compute.
{‘contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’}
"""

SIZE = 128
train_images = []
train_labels = []
label = 'A'
numClasses = 1
# #################################################
# CAMINHO CORRIGIDO AQUI
# #################################################
for directory_path in glob.glob("C:/Users/Estefani/Desktop/ClassificarMaiusculas/MAIUSCULAS"):
    while label <= 'Z':
        for img_path in glob.glob(os.path.join(directory_path, label + "001??.pgm")):
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (SIZE, SIZE))
            train_images.append(img)
            train_labels.append(label)
        label = chr(ord(label) + 1)
        numClasses = numClasses + 1
train_images = np.array(train_images)
train_labels = np.array(train_labels)
print("Treinamento Lido!!")
test_images = []
test_labels = []
label = 'A'
# #################################################
# CAMINHO CORRIGIDO AQUI
# #################################################
for directory_path in glob.glob("C:/Users/Estefani/Desktop/ClassificarMaiusculas/MAIUSCULAS"):
    while label <= 'Z':
        for img_path in glob.glob(os.path.join(directory_path, label + "002??.pgm")):
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (SIZE, SIZE))
            test_images.append(img)
            test_labels.append(label)
        label = chr(ord(label) + 1)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
print("Teste Lido!!")
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded
print("Tudo preparado!!")

def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):
        df = pd.DataFrame()
        img = dataset[image, :, :]
        GLCM = graycomatrix(img, [1], [0])
        GLCM_Energy = graycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = graycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr
        GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss
        GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom
        GLCM_contr = graycoprops(GLCM, 'contrast')[0]
        df['Contrast'] = GLCM_contr
        image_dataset = pd.concat([image_dataset, df], ignore_index=True)
    return image_dataset
    #
    #     GLCM2 = graycomatrix(img, [1], [np.pi/4])
    #     GLCM_Energy2 = graycoprops(GLCM2, 'energy')[0]
    #     df['Energy2'] = GLCM_Energy2
    #     GLCM_corr2 = graycoprops(GLCM2, 'correlation')[0]
    #     df['Corr2'] = GLCM_corr2
    #     GLCM_diss2 = graycoprops(GLCM2, 'dissimilarity')[0]
    #     df['Diss_sim2'] = GLCM_diss2
    #     GLCM_hom2 = graycoprops(GLCM2, 'homogeneity')[0]
    #     df['Homogen2'] = GLCM_hom2
    #     GLCM_contr2 = graycoprops(GLCM2, 'contrast')[0]
    #     df['Contrast2'] = GLCM_contr2
    #     GLCM3 = graycomatrix(img, [1], [np.pi/2])
    #     GLCM_Energy3 = graycoprops(GLCM3, 'energy')[0]
    #     df['Energy3'] = GLCM_Energy3
    #     GLCM_corr3 = graycoprops(GLCM3, 'correlation')[0]
    #     df['Corr3'] = GLCM_corr3
    #     GLCM_diss3 = graycoprops(GLCM3, 'dissimilarity')[0]
    #     df['Diss_sim3'] = GLCM_diss3
    #     GLCM_hom3 = graycoprops(GLCM3, 'homogeneity')[0]
    #     df['Homogen3'] = GLCM_hom3
    #     GLCM_contr3 = graycoprops(GLCM3, 'contrast')[0]
    #     df['Contrast3'] = GLCM_contr3
    #     GLCM4 = graycomatrix(img, [1], [3*np.pi/4])
    #     GLCM_Energy4 = graycoprops(GLCM4, 'energy')[0]
    #     df['Energy4'] = GLCM_Energy4
    #     GLCM_corr4 = graycoprops(GLCM4, 'correlation')[0]
    #     df['Corr4'] = GLCM_corr4
    #     GLCM_diss4 = graycoprops(GLCM4, 'dissimilarity')[0]
    #     df['Diss_sim4'] = GLCM_diss4
    #     GLCM_hom4 = graycoprops(GLCM4, 'homogeneity')[0]
    #     df['Homogen4'] = GLCM_hom4
    #     GLCM_contr4 = graycoprops(GLCM4, 'contrast')[0]
    #     df['Contrast4'] = GLCM_contr4
    #     image_dataset = pd.concat([image_dataset, df], ignore_index=True)
    # return image_dataset

image_features = feature_extractor(x_train)
print(image_features.shape)
# CORREÇÃO: Salva o arquivo na mesma pasta do script
image_features.to_csv('TreinoGLCM.txt', sep=' ', index=False)
X_for_ML = image_features
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
X_for_ML = np.reshape(image_features, (x_train.shape[0], -1))
print("Extraido dados de treinamento")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_for_ML = scaler.fit_transform(X_for_ML)
#SVM
from sklearn import svm
SVM_model = svm.SVC(decision_function_shape='ovo')
SVM_model.fit(X_for_ML, y_train)
print("treinado SVM")
#random Forest
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators=100)
RF_model.fit(X_for_ML, y_train)
#knn
from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_for_ML, y_train)
#MLP
from sklearn.neural_network import MLPClassifier
MLP_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
MLP_model.fit(X_for_ML, y_train)
#LGB
from lightgbm import LGBMClassifier
LGB_model = LGBMClassifier(n_estimators=100)
LGB_model.fit(X_for_ML, y_train)

test_features = feature_extractor(x_test)
# CORREÇÃO: Salva o arquivo na mesma pasta do script
test_features.to_csv('TesteGLCM.txt', sep=' ', index=False)
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))
test_for_RF = scaler.fit_transform(test_for_RF)
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
gerarMatrizConfusao = True
# SVM
print("Predicao SVM")
test_predictionSVM = SVM_model.predict(test_for_RF)
test_predictionSVM = le.inverse_transform(test_predictionSVM)
print("Accuracy SVM = ", metrics.accuracy_score(test_labels, test_predictionSVM))
if gerarMatrizConfusao:
    confusion_mtxTestSVM = confusion_matrix(test_labels, test_predictionSVM)
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_mtxTestSVM, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predições')
    plt.ylabel('Valores Verdadeiros')
    plt.title("SVM")
    plt.show()
# RF
print("Predicao RF")
test_predictionRF = RF_model.predict(test_for_RF)
test_predictionRF = le.inverse_transform(test_predictionRF)
print("Accuracy RF = ", metrics.accuracy_score(test_labels, test_predictionRF))
if gerarMatrizConfusao:
    confusion_mtxTestRF = confusion_matrix(test_labels, test_predictionRF)
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_mtxTestRF, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predições')
    plt.ylabel('Valores Verdadeiros')
    plt.title("RF")
    plt.show()
# KNN
print("Predicao KNN")
test_predictionKNN = KNN_model.predict(test_for_RF)
test_predictionKNN = le.inverse_transform(test_predictionKNN)
print("Accuracy KNN = ", metrics.accuracy_score(test_labels, test_predictionKNN))
if gerarMatrizConfusao:
    confusion_mtxTestKNN = confusion_matrix(test_labels, test_predictionKNN)
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_mtxTestKNN, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predições')
    plt.ylabel('Valores Verdadeiros')
    plt.title("KNN")
    plt.show()
# MLP
print("Predicao MLP")
test_predictionMLP = MLP_model.predict(test_for_RF)
test_predictionMLP = le.inverse_transform(test_predictionMLP)
print("Accuracy MLP = ", metrics.accuracy_score(test_labels, test_predictionMLP))
if gerarMatrizConfusao:
    confusion_mtxTestMLP = confusion_matrix(test_labels, test_predictionMLP)
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_mtxTestMLP, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predições')
    plt.ylabel('Valores Verdadeiros')
    plt.title("MLP")
    plt.show()
# LGB
print("Predicao LGB")
test_predictionLGB = LGB_model.predict(test_for_RF)
test_predictionLGB = le.inverse_transform(test_predictionLGB)
print("Accuracy LGB = ", metrics.accuracy_score(test_labels, test_predictionLGB))
# Calcular a matriz de confusão
if gerarMatrizConfusao:
    confusion_mtxTestLGB = confusion_matrix(test_labels, test_predictionLGB)
    plt.figure(figsize=(10,8))
    sns.heatmap(confusion_mtxTestLGB, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predições')
    plt.ylabel('Valores Verdadeiros')
    plt.title("LGB")
    plt.show()
# Check results on a few random images
import random
n = random.randint(0, x_test.shape[0] - 1)  # Select the index of image to be loaded for testing
img = x_test[n]
cv2.imshow("Teste", img)
input_img = np.expand_dims(img, axis=0)  # Expand dims so the input is (num images, x, y, c)
input_img_features = feature_extractor(input_img)
input_img_features = np.expand_dims(input_img_features, axis=0)
input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
img_prediction = SVM_model.predict(input_img_for_RF)
img_prediction = le.inverse_transform(img_prediction)  # Reverse the label encoder to original name
print("The prediction for this image is: ", img_prediction)
print("The actual label for this image is: ", test_labels[n])
cv2.waitKey(0)