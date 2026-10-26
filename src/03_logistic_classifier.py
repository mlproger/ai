import os
import numpy as np
from sklearn import linear_model

from utilities import visualize_classifier

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")

# Определение образца входных данных
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1],
              [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

for C in (1, 100):
    # Примечание: в современных версиях scikit-learn решатель 'liblinear' не
    # поддерживает мультиклассовую классификацию (>=3 классов) напрямую,
    # поэтому используется решатель 'lbfgs', который реализует ту же логику
    # логистической регрессии и поддерживает multinomial-постановку задачи.
    classifier = linear_model.LogisticRegression(solver='lbfgs', C=C, max_iter=1000)
    classifier.fit(X, y)
    out_path = os.path.join(FIG_DIR, f"03_logistic_classifier_C{C}.png")
    visualize_classifier(classifier, X, y, out_path=out_path, title=f"Logistic classifier, C={C}")
    print(f"C={C}: saved {out_path}")
    print(f"C={C}: training accuracy = {classifier.score(X, y):.4f}")
