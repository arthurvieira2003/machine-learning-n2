import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """
    Carrega o dataset de transações de cartão de crédito.
    
    Args:
        file_path: Caminho para o arquivo CSV
        
    Returns:
        DataFrame contendo os dados
    """
    print(f"Carregando dados de {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
    return df

def explore_data(df):
    """
    Realiza uma exploração inicial dos dados.
    
    Args:
        df: DataFrame com os dados
        
    Returns:
        Estatísticas básicas sobre os dados
    """
    print("\n--- Informações sobre o dataset ---")
    print(f"Forma do dataset: {df.shape}")
    
    # Verificar a distribuição da variável alvo (fraude)
    fraud_count = df['Class'].value_counts()
    print(f"\nDistribuição da classe alvo:\n{fraud_count}")
    print(f"Porcentagem de fraudes: {fraud_count[1] / len(df) * 100:.4f}%")
    
    # Verificar estatísticas básicas
    print("\nEstatísticas descritivas das features numéricas:")
    print(df.describe())
    
    # Verificar valores ausentes
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nValores ausentes:\n{missing_values[missing_values > 0]}")
    else:
        print("\nNão há valores ausentes no dataset.")
    
    return {
        'shape': df.shape,
        'fraud_ratio': fraud_count[1] / len(df),
        'missing_values': missing_values.sum()
    }

def preprocess_data(df, test_size=0.2, random_state=42, apply_smote=True):
    """
    Pré-processa os dados para treinamento.
    
    Args:
        df: DataFrame com os dados
        test_size: Proporção do conjunto de teste
        random_state: Seed para reprodutibilidade
        apply_smote: Se True, aplica SMOTE para balancear as classes
        
    Returns:
        X_train, X_test, y_train, y_test: Conjuntos de treinamento e teste
    """
    # Separando features e target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Normalização das features
    print("\nNormalizando as features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Dividindo em conjuntos de treino e teste
    print(f"Dividindo os dados (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Conjunto de teste: {X_test.shape[0]} amostras")
    
    # Aplicando SMOTE para balancear as classes no conjunto de treino
    if apply_smote:
        print("\nAplicando SMOTE para balancear as classes...")
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"Conjunto de treino após SMOTE: {X_train.shape[0]} amostras")
        print(f"Distribuição de classes após SMOTE: {np.bincount(y_train)}")
    
    return X_train, X_test, y_train, y_test 