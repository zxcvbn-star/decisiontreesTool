import plotly.graph_objects as go
from django.shortcuts import render
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model, iris, X_train, X_test, y_train, y_test


def build_context():
    pass


def home(request):
    return render(request, 'home.html', build_context())
