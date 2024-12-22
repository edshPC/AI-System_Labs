from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/california_housing_train.csv')
stats = data.describe()

print("Статистика по датасету:")
print(stats.to_string(float_format=lambda x: f'{x:.2f}'))

fig, axs = plt.subplots(4, 2, figsize=(12, 12))
axs = axs.flatten()
for i, metric in enumerate(('count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max')):
    axs[i].bar(data.columns, stats.loc[metric])
    axs[i].set_title(metric)
    axs[i].set_ylabel(metric)
    axs[i].set_xticks(np.arange(len(data.columns)))  # Устанавливаем позиции для меток
    axs[i].set_xticklabels(data.columns, rotation=45)  # Устанавливаем метки
plt.tight_layout()
plt.show()

# Обработка: Заменяем все Null на соответствующее среднее
data.fillna(value=data.mean(), inplace=True)

# Синтетический признак
data['income_value'] = data['median_income'] * data['median_house_value']

# Разделение на тренировочную и тестовую выборки
training_data = data.iloc[:16990]  # Первые 16990 значений
test_data = data.iloc[16990:]  # Оставшиеся

# Нормализация данных (минимакс)
# data_scaled = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=data.columns)

results = []

for dep in data.columns:
    # Разделение данных на признаки и целевую переменную для каждого столбца
    max_det, result = 0, ()
    indeps = data.columns.drop(dep)
    for i in range(2, len(indeps) + 1):
        for indep in combinations(indeps, i):
            X = training_data[list(indep)].to_numpy()  # Независимые
            y = training_data[dep].to_numpy()  # Зависимая

            n, k = X.shape
            N = df = n - k  # Степени свободы

            # Добавим столбец с единицами к массиву X для учета свободного члена
            X = np.column_stack((X, np.ones(n)))

            # Рассчитаем оценки наименьших квадратов для параметров
            W = np.linalg.inv(X.T @ X) @ X.T @ y

            # Рассчитаем предсказанные значения
            y_pred = X @ W.T  # Модель(X)

            y_mean = np.mean(y)  # Среднее y
            Var_data = np.sum((y - y_mean) ** 2) / N  # Дисперсия данных
            Var_reg = np.sum((y_pred - y_mean) ** 2) / N  # Регрессионная дисперсия
            R_squared = Var_reg / Var_data  # Коэффициент детерминации

            # Прогнозирование на тестовых данных
            X_test = test_data[list(indep)].to_numpy()
            X_test = np.column_stack((X_test, np.ones(X_test.shape[0])))
            y_pred_test = X_test @ W.T

            if R_squared > max_det + .005:
                max_det = R_squared
                # Обновляем результат
                result = (R_squared, dep, indep, y_pred_test, test_data[dep].to_numpy())
    results.append(result)

    print("for", dep, ":", max_det)

# Топ 3 результата по коэфициенту детерминации
top_results = sorted(results, key=lambda x: x[0], reverse=True)[:3]

for i, (r_squared, dep, indep, y_pred_test, y_real) in enumerate(top_results):
    print(f"\nМодель {i + 1} - {dep}:")
    print(f"Коэффициент детерминации: {r_squared:.4f}")
    print(f"Независимые переменные: {indep}")
    results_df = pd.DataFrame({
        'Реальные значения': y_real,
        'Предсказанные значения': y_pred_test
    })
    print(results_df)
