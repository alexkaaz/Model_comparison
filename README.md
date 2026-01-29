# Model_comparison
Model comparison
Bнструмент для автоматизированного сравнения и выбора ML-моделей в задачах бинарной классификации.

О проекте
Этот проект представляет собой комплексный ML-фреймворк, который автоматизирует процесс выбора оптимальной модели машинного обучения. Основная цель - предоставить готовое решение для быстрого проведения экспериментов с различными алгоритмами и их гиперпараметрами.

Основные возможности
Автоматическое сравнение различных алгоритмов
Полный ML-пайплайн со встроенной обработкой данных, отбор признаков, настройка гиперпараметров и оценка
Комплексная оценка метрик качества (ROC-AUC, F1, Accuracy), анализ времени выполнения и влияния параметров 
Интеграция с методами SMOTE и RandomUnderSampler для работы с несбалансированными данными
Сохранение лучших моделей и всех метрик для последующего развертывания

Технологический стек
Python
ML-библиотеки: Scikit-learn, XGBoost, imbalanced-learn
Обработка данных: Pandas
Пайплайны: Scikit-learn Pipelines, ColumnTransformer
Оптимизация: GridSearchCV с кросс-валидацией
Сериализация: Joblib для сохранения моделей

Model Comparison
A tool for automated comparison and selection of ML models in binary classification tasks.

About the Project
This project is a comprehensive ML framework that automates the process of selecting the optimal machine learning model. The main goal is to provide a ready-made solution for quickly conducting experiments with various algorithms and their hyperparameters.

Key Features
Automatic comparison of various algorithms
Complete ML pipeline with built-in data processing, feature selection, hyperparameter tuning, and evaluation
Comprehensive evaluation of quality metrics (ROC-AUC, F1, Accuracy), analysis of execution time and parameter influence
Integration with SMOTE and RandomUnderSampler methods for working with imbalanced data
Saving the best models and all metrics for subsequent deployment

Technology Stack
Python
ML Libraries: Scikit-learn, XGBoost, imbalanced-learn
Data Processing: Pandas
Pipelines: Scikit-learn Pipelines, ColumnTransformer
Optimization: GridSearchCV with cross-validation
Serialization: Joblib for saving models
