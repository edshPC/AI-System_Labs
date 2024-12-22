import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


class SimpleDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None
        self.y = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        unique_classes, counts = np.unique(y, return_counts=True)

        # все классы одинаковые
        if len(unique_classes) == 1:
            val = unique_classes[0]
            cnts = np.zeros(2)
            cnts[int(val)] = counts[0]
            return unique_classes[0], cnts
        # достигли максимальной глубины
        if depth >= self.max_depth:
            return unique_classes[np.argmax(counts)], counts

        # находим лучшее разбиение
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return unique_classes[np.argmax(counts)], counts

        # разделяем данные по признаку
        split_values = np.unique(X[:, best_feature])
        subtrees = {}
        for value in split_values:
            indices = X[:, best_feature] == value
            subtrees[value] = self._build_tree(X[indices], y[indices], depth + 1)

        return best_feature, best_threshold, subtrees

    def _best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(num_features):  # по всем признакам
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:  # по всем возможным пороговым значениям
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:  # если пустой один из массивов
                    continue

                gain = self._information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:  # если прирост информации больше чем был
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        # прирост информации как родителя минус ветви
        return self._entropy(parent) - (weight_left * self._entropy(left_child) +
                                        weight_right * self._entropy(right_child))

    def _entropy(self, y):  # Информационная энтропия Шеннона
        _, counts = np.unique(y, return_counts=True)  # количества классов
        probabilities = counts / len(y)  # вероятности
        return -np.sum(
            probabilities * np.log2(probabilities + 1e-9))  # маленькое значение для избежания логарифма нуля

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, tree):
        if len(tree) > 2:  # если не лист
            feature, threshold, subtrees = tree
            value = sample[feature]
            if value in subtrees:
                return self._predict_sample(sample, subtrees[value])
            else:
                return None  # или какое-то значение по умолчанию
        # возвращаем лист
        return tree[0]

    def predict_proba(self, X):
        return np.array([self._predict_proba_sample(sample, self.tree) for sample in X])

    def _predict_proba_sample(self, sample, tree):
        if len(tree) == 2:  # если лист
            _, class_counts = tree
            total_count = np.sum(class_counts)
            if total_count > 0:
                return class_counts[1] / total_count  # Вероятность класса 1
            else:
                return 0.5  # Если нет образцов, возвращаем 0.5

        # Если этоне лист, продолжаем рекурсию
        feature, threshold, subtrees = tree
        value = sample[feature]
        if value in subtrees:
            return self._predict_proba_sample(sample, subtrees[value])
        else:
            return 0.5  # или начение по умолчанию


data = pd.read_csv('datasets/mushrooms.csv')

# Разделение на признаки и целевую переменную, преобразования в числовые характеристики
X_all = data.drop('type', axis=1)#pd.get_dummies(data.drop('type', axis=1), drop_first=True)
print(X_all)
y = data['type'].to_numpy()  # Тип гриба: съедобный 'e', ядовитый 'p'
y = y == 'p'  # True если ядовитый

# случайные признаки
np.random.seed(42)
features_count = int(np.round(np.sqrt(X_all.shape[1])))  # sqrt(n) признаков
X_cols = np.random.choice(X_all.columns, size=features_count, replace=False)
X = X_all[X_cols].to_numpy()

# Разделение на тренировочный и тестовый
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.16, random_state=42)

model = SimpleDecisionTree(max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(f'Матрица ошибок:\n{confusion_matrix}\n')

TP = confusion_matrix.values[1, 1]  # Истинно положительные
TN = confusion_matrix.values[0, 0]  # Истинно отрицательные
FP = confusion_matrix.values[0, 1]  # Ложно положительные
FN = confusion_matrix.values[1, 0]  # Ложно отрицательные

# accuracy (точность) - доля правильно предсказанных значений
total = confusion_matrix.values.sum()  # Общее количество примеров
accuracy = (TP + TN) / total if total > 0 else 0  # точность
print(f'Accuracy: {accuracy:.2f}')

# precision (точность) - мера точности для положительных предсказаний
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
print(f'Precision: {precision:.2f}')

# recall (полнота) - мера полноты положительных предсказаний
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
print(f'Recall: {recall:.2f}')


# ROC
def compute_roc(y_true, y_scores):
    thresholds = np.linspace(0, 1, num=100)
    tpr = []
    fpr = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        tpr.append(TP / (TP + FN) if (TP + FN) > 0 else 0)  # True Positive Rate
        fpr.append(FP / (FP + TN) if (FP + TN) > 0 else 0)  # False Positive Rate

    return np.array(fpr), np.array(tpr)

# PR
def compute_pr(y_true, y_scores):
    thresholds = np.linspace(0, 1.1, num=100)
    precision = []
    recall = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        precision.append(TP / (TP + FP) if (TP + FP) > 0 else 1)
        recall.append(TP / (TP + FN) if (TP + FN) > 0 else 0)

    return np.array(precision), np.array(recall)

# Получение вероятностей предсказания
y_scores = model.predict_proba(X_test)
print(f'Уникальные вероятности: {np.unique(y_scores)}')

# Вычисление ROC и PR
fpr, tpr = compute_roc(y_test, y_scores)
precision, recall = compute_pr(y_test, y_scores)

# Площадь под кривой ROC
fpr.sort()
tpr.sort()
roc_auc = np.trapezoid(tpr, fpr)
print(f'AUC-ROC: {roc_auc:.3f}')

# Площадь под кривой PR
sorted_indices = np.argsort(recall)
recall = np.array(recall)[sorted_indices]
precision = np.array(precision)[sorted_indices]
pr_auc = np.trapezoid(precision, recall)
print(f'AUC-PR: {pr_auc:.3f}')

# График AUC-ROC
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')

# График AUC-PR
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='green', label='PR curve (area = {:.2f})'.format(pr_auc))
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()
