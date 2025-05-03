import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Avalia o desempenho do modelo usando várias métricas.
    
    Args:
        model: Modelo treinado
        X_test: Features do conjunto de teste
        y_test: Labels do conjunto de teste
        threshold: Limiar de probabilidade para classificação
        
    Returns:
        Dicionário com métricas de avaliação
    """
    print("\n--- Avaliação do Modelo ---")
    
    # Obter previsões de probabilidade e classes
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusão:")
    print(cm)
    
    # Relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Calcular AUC-ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    # Calcular Precision-Recall AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"AUC-PR: {pr_auc:.4f}")
    
    # Retornar métricas em um dicionário
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'y_proba': y_proba,
        'y_pred': y_pred
    }
    
    return metrics

def plot_confusion_matrix(cm, title='Matriz de Confusão'):
    """
    Plota a matriz de confusão.
    
    Args:
        cm: Matriz de confusão
        title: Título do gráfico
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Normal', 'Fraude'],
        yticklabels=['Normal', 'Fraude']
    )
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title(title)
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, title='Curva ROC'):
    """
    Plota a curva ROC.
    
    Args:
        fpr: Taxa de falsos positivos
        tpr: Taxa de verdadeiros positivos
        roc_auc: Área sob a curva ROC
        title: Título do gráfico
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    plt.close()

def plot_precision_recall_curve(precision, recall, pr_auc, title='Curva Precision-Recall'):
    """
    Plota a curva Precision-Recall.
    
    Args:
        precision: Valores de precisão
        recall: Valores de recall
        pr_auc: Área sob a curva Precision-Recall
        title: Título do gráfico
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'AUC = {pr_auc:.4f}')
    plt.axhline(y=sum(precision)/len(precision), color='red', linestyle='--', label='Linha Base')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.savefig('precision_recall_curve.png')
    plt.close()

def plot_feature_importance(feature_importance_df, top_n=10, title='Importância das Features'):
    """
    Plota a importância das features.
    
    Args:
        feature_importance_df: DataFrame com importância das features
        top_n: Número de features a serem exibidas
        title: Título do gráfico
    """
    if feature_importance_df is None:
        print("Não há dados de importância de features para plotar.")
        return
    
    # Selecionar as top_n features mais importantes
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Features Mais Importantes')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def find_optimal_threshold(y_test, y_proba):
    """
    Encontra o limiar ótimo para classificação baseado no score F1.
    
    Args:
        y_test: Labels verdadeiros
        y_proba: Probabilidades previstas
        
    Returns:
        Limiar ótimo
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)
    
    # Encontrar o índice do melhor limiar
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"\nLimiar ótimo encontrado: {best_threshold:.2f} (F1-Score: {best_f1:.4f})")
    
    # Plotar a relação entre limiares e F1-Score
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores, marker='o')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Limiar Ótimo: {best_threshold:.2f}')
    plt.xlabel('Limiar')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Limiar de Classificação')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_optimization.png')
    plt.close()
    
    return best_threshold 