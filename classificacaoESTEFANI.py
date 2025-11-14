import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern 
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
import time 

# Carregando os datasets de treino e teste
SIZE = 128
train_images = []
train_labels = []
label = 'A'

base_path = "C:/Users/Estefani/Desktop/ClassificarMaiusculas/MAIUSCULAS" # Caminho do cara no meu pc

print("Iniciando carregamento de dados...")
for directory_path in glob.glob(base_path):
    label = 'A' 
    while label <= 'Z':
        for img_path in glob.glob(os.path.join(directory_path, label + "001??.pgm")):
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (SIZE, SIZE))
            train_images.append(img)
            train_labels.append(label)
        label = chr(ord(label) + 1)
train_images = np.array(train_images)
train_labels = np.array(train_labels)
print(f"Treinamento Lido: {len(train_images)} imagens")

test_images = []
test_labels = []
for directory_path in glob.glob(base_path):
    label = 'A' 
    while label <= 'Z':
        for img_path in glob.glob(os.path.join(directory_path, label + "002??.pgm")):
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (SIZE, SIZE))
            test_images.append(img)
            test_labels.append(label)
        label = chr(ord(label) + 1)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
print(f"Teste Lido: {len(test_images)} imagens")

# Encoding dos Labels
le = preprocessing.LabelEncoder()
all_labels = np.concatenate([train_labels, test_labels])
le.fit(all_labels)

y_train_encoded = le.transform(train_labels)
y_test_encoded = le.transform(test_labels)
class_names = le.classes_ 

print("Labels preparados!!")

# Função Extratora de Atributos (Sem alteração aqui)
def feature_extractor(dataset, use_glcm=True, use_lbp=True, use_stats=True):
    """
    Extrai características de um conjunto de imagens com base nas flags.
    """
    image_dataset = pd.DataFrame()
    
    radius = 3
    n_points = 8 * radius
    
    for i in range(dataset.shape[0]):
        df = pd.DataFrame()
        img = dataset[i, :, :]
        
        if use_glcm:
            glcm = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
            df['GLCM_Contrast'] = [graycoprops(glcm, 'contrast').mean()]
            df['GLCM_Dissim'] = [graycoprops(glcm, 'dissimilarity').mean()]
            df['GLCM_Homogen'] = [graycoprops(glcm, 'homogeneity').mean()]
            df['GLCM_Energy'] = [graycoprops(glcm, 'energy').mean()]
            df['GLCM_Corr'] = [graycoprops(glcm, 'correlation').mean()]
            df['GLCM_ASM'] = [graycoprops(glcm, 'ASM').mean()]

        if use_lbp:
            lbp = local_binary_pattern(img, n_points, radius, method='uniform')
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, n_points + 3),
                                     range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            for j in range(len(hist)):
                df[f'LBP_hist_{j}'] = [hist[j]]

        if use_stats:
            df['Stats_Mean'] = [np.mean(img)]
            df['Stats_Std'] = [np.std(img)]
            df['Stats_Var'] = [np.var(img)]
            df['Stats_Median'] = [np.median(img)]
            # Adiciona estatísticas do Sobel (detecção de borda)
            sobel_img = sobel(img)
            df['Sobel_Mean'] = [np.mean(sobel_img)] # Mexi aqui pq deu erro
            df['Sobel_Std'] = [np.std(sobel_img)]

        image_dataset = pd.concat([image_dataset, df], ignore_index=True)
        
    return image_dataset

# O EXPERIMENTO PRINCIPAL 

# (Dicionário de classificadores foi REMOVIDO DAQUI)

feature_configs = [
    {"name": "GLCM_Only", "use_glcm": True, "use_lbp": False, "use_stats": False},
    {"name": "LBP_Only", "use_glcm": False, "use_lbp": True, "use_stats": False},
    {"name": "Stats_Only", "use_glcm": False, "use_lbp": False, "use_stats": True},
    {"name": "GLCM_ +_LBP", "use_glcm": True, "use_lbp": True, "use_stats": False},
    {"name": "GLCM_ +_Stats", "use_glcm": True, "use_lbp": False, "use_stats": True},
    {"name": "LBP_ +_Stats", "use_glcm": False, "use_lbp": True, "use_stats": True},
    {"name": "ALL_Features", "use_glcm": True, "use_lbp": True, "use_stats": True},
]

results_summary = []
best_accuracy = 0.0
best_model_name = ""
best_feature_config = ""
best_confusion_matrix = None

print("\n--- INICIANDO EXPERIMENTOS ---")

# Loop principal sobre as configurações de atributos
for config in feature_configs:
    config_name = config["name"]
    print(f"\nTestando Configuração de Atributos: {config_name}")

    classifiers = {
        "SVM": svm.SVC(decision_function_shape='ovo', kernel='rbf', C=10),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1) 
    }

    func_args = config.copy()
    del func_args['name'] 
    start_time = time.time()
    X_train_features = feature_extractor(train_images, **func_args) 
    X_test_features = feature_extractor(test_images, **func_args)
    
    print(f"  Atributos extraídos em {time.time() - start_time:.2f}s. (Shape: {X_train_features.shape})")

    # Normalização dos atributos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    results_row = {"Configuracao": config_name}

    # Loop sobre os classificadores
    for clf_name, clf in classifiers.items():
        print(f"  Treinando {clf_name}...")
        
        start_time = time.time()
        clf.fit(X_train_scaled, y_train_encoded)
        train_time = time.time() - start_time
        
        y_pred_encoded = clf.predict(X_test_scaled)
        
        accuracy = metrics.accuracy_score(y_test_encoded, y_pred_encoded)
        results_row[clf_name] = accuracy
        
        print(f"    {clf_name} Accuracy = {accuracy * 100:.2f}% (Treino: {train_time:.2f}s)")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = clf_name
            best_feature_config = config_name
            best_confusion_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)
            
    results_summary.append(results_row)

print("\nEXPERIMENTOS FINALIZADOS!")

# OS RESULTADOS

results_df = pd.DataFrame(results_summary)
results_df = results_df.set_index("Configuracao") 

print("\n### Tabela Resumo de Acurácia ###")
print(results_df.to_markdown(floatfmt=".4f"))

# SALVAR RESULTADOS PARA ANALISAR TUDO DEPOIS
try:
    csv_filename = 'resumo_resultados_classificacao.csv'
    results_df.to_csv(csv_filename)
    print(f"\n[i] Tabela de resultados salva com sucesso em: '{csv_filename}'")
except Exception as e:
    print(f"\n[!] Erro ao salvar o arquivo CSV: {e}")


print(f"\n--- Melhor Combinação ---")
print(f"Configuração de Atributos: {best_feature_config}")
print(f"Classificador: {best_model_name}")
print(f"Acurácia: {best_accuracy * 100:.2f}%")

print("\nPlotando a Matriz de Confusão do melhor modelo...")
plt.figure(figsize=(12, 10))
sns.heatmap(best_confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Melhor Modelo: {best_model_name} com {best_feature_config}")
plt.xlabel('Predições')
plt.ylabel('Valores Verdadeiros')
plt.show()

print("\nFim do script.")