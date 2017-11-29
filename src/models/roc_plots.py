from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression


X_train = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/X_train_select.csv', index_col=0)
X_test = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/X_test_select.csv', index_col=0)
y_train = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/y_train.csv', index_col=0, header=None)
y_test = pd.read_csv('/Users/meghan/DSI/capstone/Prostate-Cancer-Predictor/data/filtered/y_test.csv', index_col=0, header=None)



param_grid_ada = {'n_estimators': [100, 250, 500, 750, 1000], \
                 'learning_rate': [0.1, 0.25, 0.5, 0.75, 1.0]}
# ada1 = AdaBoostClassifier(n_estimators=5000, learning_rate=0.01)
ada2 = AdaBoostClassifier(n_estimators=10000, learning_rate=0.001)

rf1 = RandomForestClassifier(n_estimators=10000, max_features=200, max_depth=5)

sgd1 = SGDClassifier(loss='modified_huber', alpha=0.001, max_iter=100, learning_rate='optimal')
# sgd2 = SGDClassifier(loss='log', alpha=0.001, max_iter=5, learning_rate='optimal')

lg2 = LogisticRegression(penalty='l1', C=0.03)

clf_list = [ada2, rf1, sgd1, lg1, lg2]
name_list = ['Adaboost', 'Random Forest', 'Gradient Descent', 'Logistic Reg']

for e, clf in enumerate(clf_list):
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1:])
    plt.plot(fpr, tpr, lw=1, label=name_list[e])

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
