from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

DATASET_NAME = 'Iris (irysy)'


def load_split(test_size_percent, random_state):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=test_size_percent / 100.0,
        random_state=random_state,
        stratify=iris.target,
    )
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': list(iris.feature_names),
        'class_names': list(iris.target_names),
        'n_samples': len(iris.target),
    }