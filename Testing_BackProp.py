from BackPropagation import *


from sklearn import datasets

X_full, y_full = datasets.make_classification(n_samples=5000, n_features=20,
    n_informative=15,n_redundant=3,n_repeated=0,n_classes=2,flip_y=0.05,
    class_sep=1.0,shuffle=True,random_state=42)

train_test_split = int(0.8 * len(y_full))
X_train = X_full[:train_test_split, :]
y_train = y_full[:train_test_split]
X_test = X_full[train_test_split:, :]
y_test = y_full[train_test_split:]

nn = NeuralNetwork(n_hidden=[8, 3], max_iter=2000)
nn.fit(X_train, y_train)
y_hat = nn.predict(X_train)
p_hat = nn.predict_proba(X_train)
np.mean(y_train == y_hat)


nn = NeuralNetwork(n_hidden=[], max_iter=2000)
nn.fit(X_train, y_train)
y_hat = nn.predict(X_train)
p_hat = nn.predict_proba(X_train)
y_test_hat = nn.predict(X_test)
p_test_hat = nn.predict_proba(X_test)
print(np.mean(y_train == y_hat), np.mean(y_test == y_test_hat))

from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
#%matplotlib inline

fpr, tpr, threshold = roc_curve(y_test, p_test_hat)
plt.figure(figsize=(16,10))
plt.step(fpr, tpr, color='black')
plt.fill_between(fpr, tpr, step="pre", color='gray', alpha=0.2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.text(0.45, 0.55, 'AUC: {:.4f}'.format(roc_auc_score(y_test, p_test_hat)))
plt.minorticks_on()
plt.grid(True, which='both')
plt.axis([0, 1, 0, 1])
plt.show()