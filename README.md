# Projeto de Detecção de Fraudes em Cartões de Crédito

## Acadêmicos

- Arthur Henrique Tscha Vieira
- Rafael Rodrigues Ferreira de Andrade

## Descrição

Este projeto implementa um modelo de machine learning para detecção de fraudes em transações de cartão de crédito utilizando aprendizagem supervisionada. Ele aborda o problema de desbalanceamento de classes típico neste tipo de aplicação, onde a maioria das transações são legítimas e apenas uma pequena fração é fraudulenta.

O projeto utiliza o Random Forest como algoritmo principal, com opções para outros classificadores como Gradient Boosting, Regressão Logística e SVM. Para lidar com o desbalanceamento de classes, aplicamos a técnica SMOTE (Synthetic Minority Over-sampling Technique).

## Características Principais

- Pré-processamento de dados com normalização e balanceamento de classes
- Implementação de modelo de classificação Random Forest
- Otimização de hiperparâmetros (opcional)
- Métricas de avaliação específicas para dados desbalanceados (Precision, Recall, F1-Score)
- Visualizações para análise de resultados (Matriz de Confusão, Curva ROC, Precision-Recall)
- Análise de importância de features
- Otimização do limiar de classificação para equilíbrio entre precisão e recall

## Estrutura do Projeto

- `data/`: Contém o dataset de transações
  - Baixado automaticamente via script `download_dataset.py`
- `src/`: Código fonte do projeto
  - `preprocessing.py`: Funções para carregamento, exploração e pré-processamento dos dados
  - `model.py`: Implementação e treinamento dos diferentes modelos de classificação
  - `evaluation.py`: Métricas de avaliação e geração de visualizações
- `main.py`: Script principal para executar o fluxo completo
- Arquivos de visualização:
  - `confusion_matrix.png`: Matriz de confusão do modelo
  - `roc_curve.png`: Curva ROC e área sob a curva (AUC)
  - `precision_recall_curve.png`: Curva Precision-Recall
  - `feature_importance.png`: Importância das features para o modelo
  - `threshold_optimization.png`: Análise de otimização do limiar de classificação

## Requisitos

- Python 3.8+
- Bibliotecas:
  - numpy >= 1.22.0
  - pandas >= 1.5.0
  - matplotlib >= 3.5.0
  - seaborn >= 0.12.0
  - scikit-learn >= 1.0.0
  - imbalanced-learn >= 0.10.0

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/arthurvieira2003/machine-learning-n2.git
cd deteccao-fraudes
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Baixe o dataset (se necessário):

```bash
python download_dataset.py
```

## Como executar

Para executar o fluxo completo do projeto:

```bash
python main.py
```

O script irá:

1. Carregar e explorar os dados
2. Aplicar pré-processamento e balanceamento de classes
3. Treinar o modelo Random Forest
4. Avaliar o desempenho com diferentes métricas
5. Gerar visualizações para análise
6. Encontrar o limiar ótimo de classificação
7. Reavaliar o modelo com o limiar otimizado

## Resultados

O modelo gera diversas visualizações que são salvas na pasta raiz do projeto:

- Matriz de confusão para análise de falsos positivos e falsos negativos
- Curva ROC e área sob a curva (AUC) para avaliar o poder discriminatório do modelo
- Curva Precision-Recall para análise do trade-off entre precisão e cobertura
- Gráfico de importância das features para identificar quais variáveis mais contribuem para a detecção de fraudes
- Análise de otimização de limiar para encontrar o ponto de equilíbrio ideal entre precisão e recall
