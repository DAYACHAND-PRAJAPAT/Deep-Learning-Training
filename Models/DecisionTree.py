import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

def run(show_plot=True,verbose=True,random_state=42):
    data = load_breast_cancer()
    x, y = data.data, data.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    model = DecisionTreeClassifier(max_depth=5, criterion='gini',random_state=random_state)
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:,1]

    metrices = {
        "model" : "Decision Tree Classifier",
        "accuracy" : accuracy_score(y_test,y_pred),
        "f1" : f1_score(y_test,y_pred),
        "auc" : roc_auc_score(y_test,y_pred),
        "confusion_matrix" : confusion_matrix(y_test,y_pred),
        "classification_report" : classification_report(y_test,y_pred,digits=4)
    }

    if verbose:
        print("====== Decision Tree ======")
        print(f"Accuracy:{metrices['accuracy']:.4f}")
        print(f"F1 Score:{metrices['f1']:.4f}")
        print(f"Confusion Matrix:\n{metrices['confusion_matrix']}")
        print(f"Classification Report: \n{metrices['classification_report']}")
        print(f"ROC & AUC: {metrices['auc']:.4f}")

    if show_plot:
        fpr, tpr, _ = roc_curve(y_test,y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {metrices['auc']:.4f}")
        plt.plot([0,1], [0,1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("DT - ROC Curve")
        plt.legend()
        plt.savefig('DT.png')
        plt.show()

    return metrices

if __name__ == "__main__":
    run(show_plot=True,verbose=True)