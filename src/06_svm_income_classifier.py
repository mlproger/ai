import os
import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn import model_selection

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Lab01_data")

# Входной файл
input_file = os.path.join(DATA_DIR, "income_data.txt")

# Чтение данных
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line[:-1].split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Преобразование в массив numpy array
X = np.array(X)

# Преобразование строковых данных в числовые
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Создание SVM-классификатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0, dual="auto", max_iter=10000))

# Обучение классификатора
classifier.fit(X, y)

# Перекрестная проверка
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=5)
classifier = OneVsOneClassifier(LinearSVC(random_state=0, dual="auto", max_iter=10000))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# Вычисление F-меры для SVM-классификатора
f1 = model_selection.cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")

accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy on test set: " + str(round(accuracy, 2)) + "%")


def predict_income(input_data, label):
    input_data_encoded = [-1] * len(input_data)
    count = 0
    for i, item in enumerate(input_data):
        if item.isdigit():
            input_data_encoded[i] = int(input_data[i])
        else:
            input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
            count += 1

    input_data_encoded = np.array(input_data_encoded).reshape(1, -1)
    predicted_class = classifier.predict(input_data_encoded)
    result = label_encoder[-1].inverse_transform(predicted_class)[0]
    print(f"\n[{label}] input: {input_data}")
    print(f"[{label}] predicted income class: {result}")


# Тестовая точка данных из методички
predict_income(['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners',
                 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States'], "Datapoint 1 (methodology example)")

# Дополнительная точка данных: предположительно высокий доход
predict_income(['45', 'Private', '160000', 'Masters', '14', 'Married-civ-spouse', 'Exec-managerial',
                 'Husband', 'White', 'Male', '15000', '0', '50', 'United-States'], "Datapoint 2 (high income profile)")

# Дополнительная точка данных: молодой сотрудник с частичной занятостью
predict_income(['22', 'Private', '150000', 'Some-college', '10', 'Never-married', 'Sales',
                 'Own-child', 'White', 'Female', '0', '0', '20', 'United-States'], "Datapoint 3 (young part-time worker)")

# Дополнительная точка данных: самозанятый специалист средних лет
predict_income(['52', 'Self-emp-inc', '209642', 'Prof-school', '15', 'Married-civ-spouse', 'Prof-specialty',
                 'Husband', 'White', 'Male', '5178', '0', '60', 'United-States'], "Datapoint 4 (self-employed professional)")
