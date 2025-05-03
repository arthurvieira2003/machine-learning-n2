import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Importar módulos personalizados
from src.preprocessing import load_data, explore_data, preprocess_data
from src.model import train_model, get_feature_importance
from src.evaluation import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_feature_importance,
    find_optimal_threshold
)

def main():
    """
    Função principal que orquestra o fluxo de detecção de fraudes simplificado.
    """
    print("=== SISTEMA DE DETECÇÃO DE FRAUDES EM CARTÕES DE CRÉDITO (VERSÃO SIMPLIFICADA) ===\n")
    
    # Definir o caminho do dataset
    data_path = os.path.join("data", "creditcard.csv")
    
    # 1. Carregar e explorar os dados
    df = load_data(data_path)
    data_info = explore_data(df)
    
    # 2. Pré-processar os dados
    # Usamos uma amostra reduzida para tornar o processamento mais rápido
    sample_size = 50000  # Ajuste conforme necessário
    print(f"\nUsando amostra reduzida de {sample_size} transações para processamento mais rápido")
    if len(df) > sample_size:
        # Certifique-se de manter a proporção de fraudes na amostra
        fraud_df = df[df['Class'] == 1]
        non_fraud_df = df[df['Class'] == 0].sample(sample_size - len(fraud_df), random_state=42)
        df_sample = pd.concat([fraud_df, non_fraud_df])
        # Embaralhar o DataFrame
        df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        df_sample = df
    
    X_train, X_test, y_train, y_test = preprocess_data(
        df_sample, 
        test_size=0.2, 
        random_state=42,
        apply_smote=True
    )
    
    # 3. Treinar o modelo Random Forest
    print(f"\n{'='*50}")
    print(f"Processando modelo: random_forest")
    
    # Treinar o modelo com parâmetros reduzidos para execução mais rápida
    model = train_model(
        X_train, 
        y_train, 
        model_type='random_forest', 
        optimize=False,
        n_estimators=50  # Reduzir número de árvores para execução mais rápida
    )
    
    # 4. Avaliar o modelo
    metrics = evaluate_model(model, X_test, y_test)
    
    # 5. Extrair importância das features
    feature_names = df.drop('Class', axis=1).columns.tolist()
    feature_importance = get_feature_importance(
        model, 
        feature_names, 
        model_type='random_forest'
    )
    
    # 6. Plotar resultados
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        title='Matriz de Confusão - Random Forest'
    )
    
    plot_roc_curve(
        metrics['fpr'],
        metrics['tpr'],
        metrics['roc_auc'],
        title='Curva ROC - Random Forest'
    )
    
    plot_precision_recall_curve(
        metrics['precision_curve'],
        metrics['recall_curve'],
        metrics['pr_auc'],
        title='Curva Precision-Recall - Random Forest'
    )
    
    if feature_importance is not None:
        plot_feature_importance(
            feature_importance,
            top_n=10,
            title='Importância das Features - Random Forest'
        )
    
    # 7. Encontrar limiar ótimo
    optimal_threshold = find_optimal_threshold(
        y_test, 
        metrics['y_proba']
    )
    
    # 8. Reavaliar com o limiar ótimo
    print("\nReavaliando modelo com limiar ótimo:")
    metrics_optimal = evaluate_model(
        model, 
        X_test, 
        y_test, 
        threshold=optimal_threshold
    )
    
    print("\n=== ANÁLISE CONCLUÍDA ===")
    print("Os resultados foram salvos em arquivos PNG no diretório atual.")

if __name__ == "__main__":
    main() 