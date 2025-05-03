import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train, model_type='random_forest', optimize=False, n_estimators=100):
    """
    Treina um modelo de classificação para detecção de fraudes.
    
    Args:
        X_train: Features do conjunto de treinamento
        y_train: Labels do conjunto de treinamento
        model_type: Tipo de modelo a ser treinado
        optimize: Se True, aplica otimização de hiperparâmetros
        n_estimators: Número de estimadores para modelos baseados em árvores
        
    Returns:
        Modelo treinado
    """
    print(f"\n--- Treinando modelo: {model_type} ---")
    
    if model_type == 'random_forest':
        if optimize:
            print("Realizando otimização de hiperparâmetros...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            model = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=3,
                scoring='f1',
                n_jobs=-1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                class_weight='balanced',
                random_state=42
            )
    
    elif model_type == 'gradient_boosting':
        if optimize:
            print("Realizando otimização de hiperparâmetros...")
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }
            model = GridSearchCV(
                GradientBoostingClassifier(random_state=42),
                param_grid,
                cv=3,
                scoring='f1',
                n_jobs=-1
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
    
    elif model_type == 'logistic_regression':
        if optimize:
            print("Realizando otimização de hiperparâmetros...")
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced']
            }
            model = GridSearchCV(
                LogisticRegression(random_state=42, max_iter=1000),
                param_grid,
                cv=3,
                scoring='f1',
                n_jobs=-1
            )
        else:
            model = LogisticRegression(
                C=1.0,
                penalty='l2',
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
    
    elif model_type == 'svm':
        if optimize:
            print("Realizando otimização de hiperparâmetros...")
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto'],
                'class_weight': [None, 'balanced']
            }
            model = GridSearchCV(
                SVC(random_state=42, probability=True),
                param_grid,
                cv=3,
                scoring='f1',
                n_jobs=-1
            )
        else:
            model = SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
    
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    # Treinamento do modelo
    print(f"Treinando modelo com {X_train.shape[0]} amostras...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Treinamento concluído em {training_time:.2f} segundos.")
    
    # Se foi aplicada otimização, extrair o melhor modelo
    if optimize and hasattr(model, 'best_estimator_'):
        print(f"Melhores parâmetros encontrados: {model.best_params_}")
        model = model.best_estimator_
    
    return model

def get_feature_importance(model, feature_names, model_type='random_forest'):
    """
    Extrai a importância das features para modelos que suportam essa funcionalidade.
    
    Args:
        model: Modelo treinado
        feature_names: Nomes das features
        model_type: Tipo de modelo
        
    Returns:
        DataFrame com as importâncias das features
    """
    if model_type in ['random_forest', 'gradient_boosting']:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Criar DataFrame para facilitar a visualização
        feature_importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': [importances[i] for i in indices]
        })
        
        return feature_importance_df
    
    elif model_type == 'logistic_regression':
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
            indices = np.argsort(importances)[::-1]
            
            feature_importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in indices],
                'Importance': [importances[i] for i in indices]
            })
            
            return feature_importance_df
    
    print(f"Extração de importância de features não suportada para o modelo: {model_type}")
    return None 