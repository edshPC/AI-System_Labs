import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('datasets/WineDataset.csv')
stats = data.describe()

print("Статистика по датасету:")
print(stats.to_string(float_format=lambda x: f'{x:.2f}'))

data.hist(bins=50, figsize=(20, 12))
plt.show()

# Обработка: Заменяем все Null на соответствующее среднее
data.fillna(value=data.mean(), inplace=True)

# Разделение на признаки и целевую переменную
X_unscaled = data.drop('Wine', axis=1)
y = data['Wine']

# Масштабирование
X = pd.DataFrame(StandardScaler().fit_transform(X_unscaled), columns=X_unscaled.columns)

# Разделение на тренировочный и тестовый
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# расстояние между точками
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# реализация К ближайших соседей
def knn(X_train, y_train, X_test, k):
    y_pred = []
    for test_point in X_test:
        # расстояния от тестовой точки до всех точек в тренировочных данных
        distances = np.array([euclidean_distance(test_point, x) for x in X_train])

        # индексы k ближайших соседей (сортированных во возрастанию расстояний)
        k_idxs = distances.argsort()[:k]

        # Получаем метки классов для k ближайших соседей
        k_nearest_labels = y_train[k_idxs]

        # print(k_idxs, k_nearest_labels)
        # Определяем наиболее частую метку среди соседей
        most_common = np.bincount(k_nearest_labels).argmax()
        y_pred.append(most_common)

    return np.array(y_pred)


def evaluate_model(y_test, y_pred):
    # матрица ошибок
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix)

    # точность
    TP = confusion_matrix.values.diagonal().sum()  # Сумма правильных предсказаний
    total = confusion_matrix.values.sum()  # Общее количество примеров
    accuracy = TP / total if total > 0 else 0  # точность
    print(f'Accuracy: {accuracy:.2f}')

# случайные признаки
np.random.seed(42)
random_features = np.random.choice(X_train.columns, size=3, replace=False)
y_train = y_train.to_numpy()

# Обучение модели 1
X_train_random = X_train[random_features].to_numpy()
X_test_random = X_test[random_features].to_numpy()

# Прогнозирование
for k in [1, 3, 5, 10, 50]:
    y_pred_random = knn(X_train_random, y_train, X_test_random, k)
    print(f'\nМатрица ошибок для k={k} (случайные признаки):')
    evaluate_model(y_test, y_pred_random)

# фиксированные признаки (первые 3)
fixed_features = X_train.columns[:3]

# Обучение модели 2
X_train_fixed = X_train[fixed_features].to_numpy()
X_test_fixed = X_test[fixed_features].to_numpy()

# Прогнозирование
for k in [1, 3, 5, 10, 50]:
    y_pred_fixed = knn(X_train_fixed, y_train, X_test_fixed, k)
    print(f'\nМатрица ошибок для k={k} (фиксированные признаки):')
    evaluate_model(y_test, y_pred_random)

# 3D-визуализация
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*(data[fixed_features].values.transpose()), c=data['Wine'])
ax.set_xlabel(fixed_features[0])
ax.set_ylabel(fixed_features[1])
ax.set_zlabel(fixed_features[2])
plt.show()
