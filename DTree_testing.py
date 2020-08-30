from DecisionTree import *
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt


breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

model = DecisionTreeClassifier(max_depth=4)
model.fit(X, y)
y_hat = model.predict(X)
p_hat = model.predict_proba(X)[:,1]
print(confusion_matrix(y, y_hat))
print('Accuracy:', accuracy_score(y, y_hat))

fpr, tpr, threshold = roc_curve(y, p_hat)
plt.figure(figsize=(16,10))
plt.step(fpr, tpr, color='black')
plt.fill_between(fpr, tpr, step="pre", color='gray', alpha=0.2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.text(0.45, 0.55, 'AUC: {:.4f}'.format(roc_auc_score(y, 	p_hat)))
plt.minorticks_on()
plt.grid(True, which='both')
plt.axis([0, 1, 0, 1])
plt.show()