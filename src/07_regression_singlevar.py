import os
import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Lab01_data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

input_file = os.path.join(DATA_DIR, "data_singlevar_regr.txt")

# Загрузка данных
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Разбивка данных на обучающий и тестовый наборы
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Создание и обучение объекта линейного регрессора
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Прогнозирование результата
y_test_pred = regressor.predict(X_test)

# Построение графика
plt.figure()
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title("Single variable linear regression")
plt.savefig(os.path.join(FIG_DIR, "07_singlevar_regression.png"), dpi=150, bbox_inches="tight")
plt.close()

# Вычисление метрик
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Сохранение модели
output_model_file = os.path.join(MODEL_DIR, "model_singlevar.pkl")
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# Загрузка модели и повторный прогноз
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

y_test_pred_new = regressor_model.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
