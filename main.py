import numpy as np
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Classification Datasets")


def get_dataset(dataset_name_):
    dataset_dict = {
        'Iris': datasets.load_iris(),
        'Breast Cancer': datasets.load_breast_cancer(),
        "Wine Dataset": datasets.load_wine(),
    }
    X_ = dataset_dict[dataset_name_].data
    y_ = dataset_dict[dataset_name_].target
    return X_, y_


def add_parameter_ui(clf_name):
    params = {}
    if clf_name == 'KNN':
        k = st.sidebar.slider('Number of Neighbours (K)', 1, 20)
        params['KNN'] = k

    elif clf_name == 'SVM':
        c = st.sidebar.slider('Regularization parameter (C)', 0.01, 10.0)
        k = st.sidebar.selectbox('Kernel (k)', ('rbf', 'linear', 'poly', 'sigmoid'))
        d = st.sidebar.slider('Degree (d)', 1, 5)
        params['Degree'] = d
        params['Kernel'] = k
        params['C'] = c

    elif clf_name == 'Random Forest':
        max_depth = st.sidebar.slider('Max Depth (D)', 2, 15)
        n_estimators = st.sidebar.slider('Number of Estimators (e)', 1, 100)
        params['Max Depth'] = max_depth
        params['Number of Estimators'] = n_estimators

    elif clf_name == 'Naive Bayes':
        var_smoothing = st.sidebar.slider('Variance Smoothing', 0.000000001, 0.0000001)
        params['Variance Smoothing'] = var_smoothing
    return params


def get_classifier(clf_name, params):
    if clf_name == 'KNN':
        return KNeighborsClassifier(n_neighbors=params['KNN'])

    elif clf_name == 'SVM':
        return SVC(C=params['C'], kernel=params['Kernel'], degree=params['Degree'])

    elif clf_name == 'Random Forest':
        return RandomForestClassifier(n_estimators=params['Number of Estimators'], max_depth=params['Max Depth'])

    elif clf_name == 'Naive Bayes':
        return GaussianNB(var_smoothing=params['Variance Smoothing'])
    return None


def graphing_dataset(X_, y_):
    fig_ = plt.figure()
    plt.scatter(X_[:, 0], X_[:, 1], c=y_, alpha=0.8, cmap='viridis')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Dataset')
    return fig_

plt.style.use('dark_background')
dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', "Wine Dataset"))
classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Random Forest', 'Naive Bayes'))

X, y = get_dataset(dataset_name)

st.write("Shape of Dataset: ", X.shape)
st.write("Number of Classes: ", len(np.unique(y)))

# graphing the dataset
st.pyplot(graphing_dataset(X, y))

clf_params = add_parameter_ui(classifier_name)
clf = get_classifier(classifier_name, clf_params)

# Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, average='micro'),
    'Recall': recall_score(y_test, y_pred, average='micro'),
    'F1 Score': f1_score(y_test, y_pred, average='micro')
}

st.write(f"We are using the {classifier_name} classifier to classify the {dataset_name} dataset.")
st.table(metrics)

# Plotting
st.header("Plotting")
st.write("Plotting the dataset using PCA")
pca = PCA(2)
X_projected = pca.fit_transform(X)
fig = plt.figure()
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, alpha=0.8, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA')
plt.colorbar()
st.pyplot(fig)
