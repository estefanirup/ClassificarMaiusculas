# 🔤 Classificação de Letras Maiúsculas com Machine Learning

Projeto de **classificação de imagens** de letras maiúsculas (A-Z) utilizando extração manual de características e comparação de múltiplos algoritmos de Machine Learning.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

## 📋 Objetivo

Investigar qual combinação de **descritores de textura** e **classificadores** produz a melhor acurácia na tarefa de reconhecimento de letras manuscritas, considerando também o trade-off entre tempo de processamento e performance.

## 🔬 Metodologia

### Extração de Características

Foram implementados três tipos de descritores:

| Descritor | Descrição | Nº de Features |
|-----------|-----------|----------------|
| **GLCM** | Gray-Level Co-occurrence Matrix - captura padrões de textura baseados em relações espaciais entre pixels | 6 |
| **LBP** | Local Binary Pattern - descreve padrões locais de textura | 26 |
| **Stats** | Estatísticas básicas (média, desvio padrão, variância, mediana) + Sobel | 6 |

### Classificadores Avaliados

- **SVM** (Support Vector Machine) com kernel RBF
- **Random Forest** (100 estimadores)
- **KNN** (K-Nearest Neighbors, k=5)
- **MLP** (Multi-Layer Perceptron)
- **LightGBM** (Gradient Boosting)

### Pipeline de Processamento

```
Imagem → Resize (128x128) → Extração de Features → Normalização (StandardScaler) → Classificação
```

## 📊 Resultados

### Tabela de Acurácia por Configuração

| Configuração | SVM | Random Forest | KNN | MLP | LightGBM |
|--------------|-----|---------------|-----|-----|----------|
| GLCM Only | 88.62% | 84.81% | 84.54% | 77.38% | 86.88% |
| LBP Only | 95.31% | 94.00% | 90.00% | 92.96% | 93.73% |
| Stats Only | 77.35% | 79.23% | 74.92% | 72.77% | 79.54% |
| GLCM + LBP | 96.27% | 95.69% | 91.08% | 94.85% | 95.27% |
| GLCM + Stats | 89.77% | 85.96% | 84.77% | 87.42% | 88.54% |
| LBP + Stats | 96.50% | 95.69% | 90.85% | 94.35% | 94.50% |
| **ALL Features** | **96.65%** | 95.27% | 91.81% | 95.38% | 94.88% |

### 🏆 Melhor Resultado

- **Configuração:** ALL Features (GLCM + LBP + Stats)
- **Classificador:** SVM
- **Acurácia:** 96.65%

### ⚡ Melhor Custo-Benefício

| Configuração | Tempo de Extração | Acurácia |
|--------------|-------------------|----------|
| ALL Features | 376.63s | 96.65% |
| **LBP + Stats** | **97.02s** | **96.50%** |

> **Insight:** A configuração LBP + Stats é **4x mais rápida** com apenas **0.15% menos acurácia** — ideal para sistemas em tempo real.

## 💡 Principais Conclusões

1. **LBP foi o descritor mais impactante**, alcançando sozinho 95.31% de acurácia
2. **Combinar descritores sempre melhorou os resultados**
3. **SVM foi o melhor classificador** em 6 das 7 configurações
4. **Estatísticas simples agregaram valor** quando combinadas com outros descritores
5. **KNN teve o pior desempenho** apesar de ser o mais rápido para treinar

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **scikit-learn** - Classificadores e métricas
- **scikit-image** - Extração de features (GLCM, LBP)
- **OpenCV** - Processamento de imagens
- **LightGBM** - Gradient Boosting
- **pandas/NumPy** - Manipulação de dados
- **Matplotlib/Seaborn** - Visualização

