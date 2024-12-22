from itertools import product

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

train_data = pd.read_csv('datasets/titanic/train.csv')
test_data = pd.read_csv('datasets/titanic/test.csv')
test_answers = pd.read_csv('datasets/titanic/gender_submission.csv')

# удаляем неинформативные столбцы
train_data.drop(['PassengerId', 'Name' ,'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# male -> 0, female -> 1 (перевод в числовые значения
train_data['Sex'], _ = train_data['Sex'].factorize()
train_data['Embarked'], _ = train_data['Embarked'].factorize()
test_data['Sex'], _ = test_data['Sex'].factorize()
test_data['Embarked'], _ = test_data['Embarked'].factorize()

stats = train_data.describe()

print("Статистика по датасету:")
print(stats.to_string(float_format=lambda x: f'{x:.2f}'))

fig, axs = plt.subplots(4, 2, figsize=(12, 12))
axs = axs.flatten()
for i, metric in enumerate(('count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max')):
    axs[i].bar(train_data.columns, stats.loc[metric])
    axs[i].set_title(metric)
    axs[i].set_ylabel(metric)
    axs[i].set_xticks(np.arange(len(train_data.columns)))  # Устанавливаем позиции для меток
    axs[i].set_xticklabels(train_data.columns, rotation=45)  # Устанавливаем метки
plt.tight_layout()
plt.show()

# Обработка: Заменяем все Null на соответствующее среднее
train_data.fillna(value=train_data.mean(), inplace=True)
test_data.fillna(value=test_data.mean(), inplace=True)

# Разделяем на признаки и целевую
X_train = train_data.drop('Survived', axis=1).values
y_train = train_data['Survived'].values
X_test = test_data.values
y_test = test_answers['Survived'].values


class LogisticRegression:
    def __init__(self):
        self.w = []
        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def log_loss(self, y_true, y_pred):
        # обрезаем в допустимый диапазон
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # логарифмическая потеря
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y, iterations=20, learning_rate=0.01):
        m, n = X.shape  # m объектов, n признаков
        self.w = np.zeros(n)  # веса
        self.b = 0  # смещение

        for i in range(iterations):
            # 1. Вычисляем предсказания модели
            z = X @ self.w + self.b
            y_pred = self.sigmoid(z)

            self.update_model(X, y, y_pred, learning_rate)

            # Выводим значение функции потерь (1 и посл. итерация)
            if i == 0 or i == iterations-1:
                loss = self.log_loss(y, y_pred)
                print(f"Итерация {i}, Функция потерь: {loss:.4f}")

    def update_model(self, X, y, y_pred, learning_rate):
        pass

    def predict(self, X):
        # умножаем признаки на веса, добавляем смещение.
        z = X @ self.w + self.b
        # результат -> вероятность
        y_pred = self.sigmoid(z)
        # если результат больше 50%, считаем за истину
        return (y_pred > 0.5).astype(int)


class GradientDescentLogisticRegression(LogisticRegression):
    def update_model(self, X, y, y_pred, learning_rate):
        m, n = X.shape
        # Вычисляем градиенты
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # Обновляем веса и смещение
        self.w -= learning_rate * dw
        self.b -= learning_rate * db

class NewtonLogisticRegression(LogisticRegression):
    def update_model(self, X, y, y_pred, _):
        m, n = X.shape  # m объектов, n признаков

        # Вычисляем градиенты
        dw = (1 / m) * np.dot(X.T, (y_pred - y))  # Градиент для весов
        db = (1 / m) * np.sum(y_pred - y)  # Градиент для смещения

        # Вычисляем Гессиан (матрица второй производной)
        R = np.diag(y_pred * (1 - y_pred))  # Диагональная матрица (m x m)
        H_w = (1 / m) * np.dot(np.dot(X.T, R), X)  # Гессиан для весов
        H_b = np.sum(y_pred * (1 - y_pred)) / m  # Гессиан для смещения (скаляр)

        # Обновляем веса и смещение
        self.w -= np.linalg.solve(H_w, dw)  # Обратная матрица Гессе для весов
        self.b -= db / H_b  # Обновление смещения



def calcualte_params(model: LogisticRegression, iterations_list, learning_rates=(None,)):
    for (iterations, rate) in product(iterations_list, learning_rates):
        model.fit(X_train, y_train, iterations, rate)
        y_pred = model.predict(X_test)

        confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        confusion_matrix = confusion_matrix.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        # print(confusion_matrix)
        TP = confusion_matrix.values[1, 1]  # Истинно положительные
        TN = confusion_matrix.values[0, 0]  # Истинно отрицательные
        FP = confusion_matrix.values[0, 1]  # Ложно положительные
        FN = confusion_matrix.values[1, 0]  # Ложно отрицательные

        # accuracy (точность) - доля правильно предсказанных значений
        total = confusion_matrix.values.sum()  # Общее количество примеров
        accuracy = (TP + TN) / total if total > 0 else 0  # точность
        # precision (точность) - мера точности для положительных предсказаний
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        # recall (полнота) - мера полноты положительных предсказаний
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        # f1_score - гармоническое среднее между precision и recall
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f'{iterations}, {rate}: {accuracy:.2f} / {precision:.2f} / {recall:.2f} / {f1_score:.2f}')


grad = GradientDescentLogisticRegression()
newton = NewtonLogisticRegression()

print("Iterations, Learning rate: Accuracy / Precision / Recall / F1 Score")

print("Реализация методом градиетного спуска:")
calcualte_params(grad, [10, 100, 250, 500, 1000], [1, 0.1, 0.01, 0.001, 0.0001])
print("\nРеализация методом оптимизации Ньютона:")
calcualte_params(newton, [10, 100, 250, 500, 1000])

