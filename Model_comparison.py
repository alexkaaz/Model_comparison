import pandas as pd
import numpy as np
import joblib
import time
from typing import Tuple, List
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore")

def get_columns_by_dtype(dataset: pd.DataFrame) -> Tuple[List[str], List[str]]:
    '''
    Возвращает списки столбцов разделенных по типам данных
    '''
    numeric = ['int', 'int32', 'int64', 'float', 'float32', 'float64']
    categories = ['object', 'category', 'bool']
    num_cols = dataset.select_dtypes(include=numeric).columns.tolist()
    nan_cols = dataset.select_dtypes(include=categories).columns.tolist()
    return num_cols, nan_cols

def best_model_search(features: pd.DataFrame, target: pd.Series, model,  
        search_params: dict, cols: Tuple[List[str], List[str]]) -> GridSearchCV:
    '''
    Полный цикл подбора гиперпараметров модели с помощью GridSearchCV(). 
    Подбор совершается для модели model по паратрам search_params

    model: любая модель, для которой возможен подбор параметров с GridSearchCV()
    search_params: возможные параметры модели. Записываются с префиксом <model__> 
    cols: кортеж столбцов по типам данных 
    '''
    # Анализ дисбаланса классов
    print('='*50)
    print(f"Распределение классов:")
    print(target.value_counts())
    print(f"Соотношение классов: {target.mean():.2%} положительных")

    # Для поиска лучших параметров
    selector = SelectFromModel(
        Lasso(alpha=0.1), 
        threshold='median'
    )

    # Разделение на признаки и целевую переменную
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, 
        random_state=42, 
        shuffle=True, 
        test_size=0.2,
        stratify=target 
    )

    # Пайплайн для числовых признаков
    pipe_num = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])

    # Пайплайн для категориальных признаков
    pipe_cat = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])

    num_cols, nan_cols = cols

    col_transformer = ColumnTransformer([
        ('numeric', pipe_num, num_cols),
        ('categorical', pipe_cat, nan_cols)
    ])

    # Финальный пайплайн
    final_pipe = Pipeline([
        ('prep', col_transformer),
        ('feature_selection', selector), 
        ('model', model)
    ])

    # Подбор гиперпараметров
    grid_search = GridSearchCV(
        final_pipe, 
        search_params, 
        cv=5,  
        scoring='roc_auc',  
        n_jobs=-1,
        verbose=1
    )

    rf_time = time.time()
    print('='*50)
    print(f"Подбор гиперпараметров")
    grid_search.fit(X_train, y_train)
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучший ROC-AUC на кросс-валидации: {grid_search.best_score_:.4f}")
    print('='*50)

    # Оценка лучшей модели
    best_model = grid_search.best_estimator_

    # Предсказания
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Метрики
    print(f"RESULT OF GRIDSEARCHCV")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Время обучения и предсказания GreadSearchCV: {time.time() - rf_time}")
    print('='*50)

    return best_model

def best_features(model: GridSearchCV, cols) -> None:
    '''
    Выводит лучшие параметры в порядке убывания значимости 
    '''
    # Важность каждого параметра
    importances = model.named_steps['model'].feature_importances_

    # Не нулевые параметры
    features_ = model.named_steps['feature_selection']
    collect_mask = features_.get_support()

    # Названия столбцов после обработки 
    preproc = model.named_steps['prep']
    cat_e = preproc.named_transformers_['categorical'].named_steps['encoder']
    cat_f = cat_e.get_feature_names_out().tolist()
    
    num_s = preproc.named_transformers_['numeric'].named_steps['scaler']
    num_f = num_s.get_feature_names_out().tolist()

    all_f = num_f + cat_f
    selected_f = np.array(all_f)[collect_mask].tolist()
    
    # Вывод значимости каждого параметра
    print(pd.DataFrame({
        'Признаки': selected_f,
        'Значимость': importances
    }).sort_values('Значимость', ascending=False))
