import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Чтение данных
df = pd.read_csv('merged_sh_py3.csv')

# Кодирование категориальных фичей с помощью OneHotEncoder
columns_to_encode = [
    'event_category',
    'utm_source',
    'utm_medium',
    'utm_campaign',
    'utm_adcontent',
]

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_data = encoder.fit_transform(df[columns_to_encode])

# Получить имена закодированных столбцов
encoded_columns = encoder.get_feature_names_out(columns_to_encode)

# Создать новый DataFrame с закодированными признаками
df_encoded = pd.concat([df, pd.DataFrame(encoded_data, columns=encoded_columns)], axis=1)

# Удаление ненужных столбцов
columns_to_drop = [
    'client_id',
    'hit_number',
    'hit_page_path',
    'event_category',
    'visit_number',
    'utm_source',
    'utm_medium',
    'utm_campaign',
    'utm_adcontent',
    'device_category',
    'device_screen_resolution',
    'device_browser',
    'geo_country',
    'geo_city',
    'visit_datetime'
]

df_encoded.drop(columns_to_drop, axis=1, inplace=True)

# Разделение данных на признаки и целевую переменную
X = df_encoded.drop(['event_action_new'], axis=1)
y = df_encoded['event_action_new']

# Определение моделей
models = [
    ('Logistic Regression', LogisticRegression(solver='liblinear')),
    ('Random Forest', RandomForestClassifier()),
    ('MLP', MLPClassifier(activation='logistic', hidden_layer_sizes=(256, 128, 64)))
]

# Создание пайплайна
best_score = 0
best_pipe = None
best_model_name = None

for name, model in models:
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # Масштабирование числовых признаков
        ('model', model)               # Обучение модели
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy', n_jobs=1)
    average_score = np.mean(scores)
    print(f"Model: {name}, Average Accuracy: {average_score:.4f}")
    if average_score > best_score:
        best_score = average_score
        best_pipe = pipe
        best_model_name = name

print(f"Best model: {best_model_name}, best accuracy: {best_score}")