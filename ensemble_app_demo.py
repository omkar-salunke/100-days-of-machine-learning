import streamlit as st
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
# Sidebar options
dataset_choice = st.sidebar.selectbox("Select Dataset", ["U-Shaped", "Linear"])
estimators_choice = st.sidebar.multiselect("Select Estimators", 
    ["Logistic Regression", "KNN", "Gaussian Naive Bayes", "SVM", "Random Forest"],
    default=["Logistic Regression", "KNN", "Gaussian Naive Bayes"]
)

# Load dataset
dataset_choice = st.sidebar.selectbox("Select Dataset", ["U-Shaped", "Linear", "Circles", "Blobs"]) 
if dataset_choice == "U-Shaped": 
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42) 
elif dataset_choice == "Linear":
    X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, random_state=42) 
elif dataset_choice == "Circles": 
    X, y = make_circles(n_samples=300, noise=0.2, factor=0.5, random_state=42) 
else: 
    X, y = make_blobs(n_samples=300, centers=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build estimators
estimators = []
if "Logistic Regression" in estimators_choice:
    estimators.append(('lr', LogisticRegression()))
if "KNN" in estimators_choice:
    estimators.append(('knn', KNeighborsClassifier()))
if "Gaussian Naive Bayes" in estimators_choice:
    estimators.append(('gnb', GaussianNB()))
if "SVM" in estimators_choice:
    estimators.append(('svm', SVC(probability=True)))
if "Random Forest" in estimators_choice:
    estimators.append(('rf', RandomForestClassifier()))

# Voting Classifier
voting_clf = VotingClassifier(estimators=estimators, voting='soft')
voting_clf.fit(X_train, y_train)

# Accuracy
st.subheader("Classification Metrics")
st.write(f"Voting Classifier Accuracy: {voting_clf.score(X_test, y_test):.2f}")
for name, model in estimators:
    model.fit(X_train, y_train)
    st.write(f"Accuracy for {name}: {model.score(X_test, y_test):.2f}")

# Plot decision boundary
def plot_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", edgecolor="k")
    st.pyplot(plt)

st.subheader("Decision Boundary")
plot_decision_boundary(voting_clf, X_test, y_test)

# streamlit run app.py
# streamlit run C:\Users\ossal\Documents\pythonProject\py_proj\100-days-of-machine-learning\ensemble_app_demo.py