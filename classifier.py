import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score

with h5py.File('data.h5', 'r') as hdf:
    x_train = hdf['dataset/train/train_dataset'][:, 1:]
    y_train = hdf['dataset/train/train_dataset'][:, 0]
    x_test = hdf['dataset/test/test_dataset'][:, 1:]
    y_test = hdf['dataset/test/test_dataset'][:, 0]


clf = make_pipeline(LogisticRegression(max_iter=1000))
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
y_clf_prob = clf.predict_proba(x_test)
print('y_pred is:', y_pred)
print('y_clf_prob is:', y_clf_prob)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy", accuracy)

recall = recall_score(y_test, y_pred)
print('recall is: ', recall)

cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['walking', 'jumping'])
cmd.plot()
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_auc = roc_auc_score(y_test, y_clf_prob[:, 1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Logistic Regression').plot()
plt.title("ROC Curve")
plt.show()

print('AUC is:', roc_auc)
