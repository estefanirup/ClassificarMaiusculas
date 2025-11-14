import pandas as pd
import matplotlib.pyplot as plt

# 1. Dados dos seus experimentos
# (Eu copiei e colei os resultados da sua execução)
data = {
    "GLCM_Only": {
        "SVM": 0.8862, "RandomForest": 0.8481, "KNN": 0.8454, "MLP": 0.7738, "LightGBM": 0.8688
    },
    "LBP_Only": {
        "SVM": 0.9531, "RandomForest": 0.9400, "KNN": 0.9000, "MLP": 0.9296, "LightGBM": 0.9373
    },
    "Stats_Only": {
        "SVM": 0.7735, "RandomForest": 0.7923, "KNN": 0.7492, "MLP": 0.7277, "LightGBM": 0.7954
    },
    "GLCM_ +_LBP": {
        "SVM": 0.9627, "RandomForest": 0.9569, "KNN": 0.9108, "MLP": 0.9485, "LightGBM": 0.9527
    },
    "GLCM_ +_Stats": {
        "SVM": 0.8977, "RandomForest": 0.8596, "KNN": 0.8477, "MLP": 0.8742, "LightGBM": 0.8854
    },
    "LBP_ +_Stats": {
        "SVM": 0.9650, "RandomForest": 0.9569, "KNN": 0.9085, "MLP": 0.9435, "LightGBM": 0.9450
    },
    "ALL_Features": {
        "SVM": 0.9665, "RandomForest": 0.9527, "KNN": 0.9181, "MLP": 0.9538, "LightGBM": 0.9488
    }
}

# 2. Criar o DataFrame
# Usamos orient='index' para que as chaves do dicionário (configurações) virem as linhas
df = pd.DataFrame.from_dict(data, orient='index')

# 3. Gerar o Gráfico
plt.figure(figsize=(16, 9))  # Tamanho da figura (largura, altura)
df.plot(
    kind='bar',  # Tipo: gráfico de barras
    figsize=(16, 9),
    fontsize=12
)

# 4. Ajustes e Títulos
plt.title("Comparação de Acurácia: Classificadores vs. Configurações de Atributos", fontsize=16)
plt.ylabel("Acurácia", fontsize=14)
plt.xlabel("Configuração de Atributos", fontsize=14)

# Ajuste crucial: Define o limite do eixo Y
# Como o pior resultado foi ~72%, começamos em 0.70 para ver melhor as diferenças
plt.ylim(0.70, 1.0) 

# Rotaciona os labels do eixo X para caberem
plt.xticks(rotation=30, ha='right')

# Adiciona uma grade horizontal para facilitar a leitura
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Posiciona a legenda
plt.legend(title="Classificadores", fontsize=12, title_fontsize=13, loc='upper left')

# Ajusta o layout para não cortar os labels
plt.tight_layout()

# 5. Mostrar o gráfico
print("Gerando gráfico...")
plt.show()
print("Gráfico exibido.")