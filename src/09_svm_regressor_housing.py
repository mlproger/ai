import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# ПРИМЕЧАНИЕ: классический датасет sklearn.datasets.load_boston() удален из
# scikit-learn, начиная с версии 1.2, по этическим соображениям (один из
# признаков датасета был построен на основе расовой сегрегации). Официальная
# рекомендованная замена от разработчиков sklearn - California Housing
# dataset, которую мы используем ниже. Структура кода и логика эксперимента
# сохранены в соответствии с методичкой.
data = fetch_california_housing()
print("Feature names:", data.feature_names)
print("Full dataset shape:", data.data.shape)

# Перемешивание данных
X, y = shuffle(data.data, data.target, random_state=7)

# SVR с линейным ядром масштабируется квадратично/кубически по числу
# примеров, поэтому для адекватного времени обучения (исходный Boston
# housing dataset содержал всего 506 точек) ограничиваем выборку до 600
# случайных наблюдений вместо полных 20 640.
SAMPLE_SIZE = 600
X, y = X[:SAMPLE_SIZE], y[:SAMPLE_SIZE]
print("Working sample shape:", X.shape)

# Разбивка данных на обучающий и тестовый наборы
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Признаки California Housing имеют очень разный масштаб (Population -
# тысячи, Latitude/Longitude - десятки, AveOccup - единицы), из-за чего
# решатель SVR сходится крайне медленно. Поэтому, в отличие от оригинальной
# методички (где Boston housing был уже в сопоставимых масштабах), здесь
# дополнительно применяется стандартизация признаков.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def run_svr(C, epsilon, label):
    sv_regressor = SVR(kernel='linear', C=C, epsilon=epsilon)
    sv_regressor.fit(X_train_scaled, y_train)

    y_test_pred = sv_regressor.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_test_pred)
    evs = explained_variance_score(y_test, y_test_pred)

    print(f"\n#### Performance [{label}] (C={C}, epsilon={epsilon}) ####")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))

    return sv_regressor


# Базовая конфигурация из методички
base_regressor = run_svr(C=1.0, epsilon=0.1, label="base")

test_points = [
    ("Point 1 (low median income block)", [1.5, 25.0, 4.0, 1.0, 900.0, 2.8, 34.1, -118.3]),
    ("Point 2 (average block)", [3.8, 30.0, 5.5, 1.05, 1400.0, 3.0, 36.5, -119.7]),
    ("Point 3 (high median income block)", [8.0, 15.0, 7.2, 1.1, 1000.0, 2.5, 37.8, -122.4]),
    ("Point 4 (older, denser block)", [2.4, 45.0, 4.8, 1.2, 2000.0, 3.5, 34.0, -118.2]),
]

print("\n--- Predictions with base model (C=1.0, epsilon=0.1) ---")
for label, point in test_points:
    point_scaled = scaler.transform([point])
    pred = base_regressor.predict(point_scaled)[0]
    print(f"{label}: predicted median house value = {round(pred, 3)} (x100 000 $)")

# Эксперименты с различными значениями C и epsilon (не менее трех комбинаций)
param_grid = [
    (0.1, 0.1),
    (1.0, 0.5),
    (10.0, 0.1),
    (10.0, 0.01),
]

print("\n--- Experiments with different C and epsilon ---")
for C, epsilon in param_grid:
    regressor = run_svr(C=C, epsilon=epsilon, label=f"C={C}, epsilon={epsilon}")
    for label, point in test_points[:2]:
        point_scaled = scaler.transform([point])
        pred = regressor.predict(point_scaled)[0]
        print(f"  {label}: predicted = {round(pred, 3)} (x100 000 $)")
