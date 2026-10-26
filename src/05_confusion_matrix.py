import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")

# Определение выборочных меток
true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

# Построение матрицы неточностей
confusion_mat = confusion_matrix(true_labels, pred_labels)
print("\nConfusion matrix:\n", confusion_mat)

# Визуализация матрицы неточностей
plt.figure()
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.savefig(os.path.join(FIG_DIR, "05_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

# Отчет о результатах классификации
targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
print('\n', classification_report(true_labels, pred_labels, target_names=targets))
