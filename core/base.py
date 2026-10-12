import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from io import StringIO
from sklearn.tree import DecisionTreeClassifier, plot_tree


def train_tree_for_template(
    X_train,
    y_train,
    feature_names,
    class_names,
    max_depth=5,
    criterion="gini",
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
    plot_tree(
        model,
        feature_names=list(feature_names),
        class_names=list(class_names),
        filled=True,
        rounded=True,
        proportion=False,
        precision=3,
        ax=ax
    )
    ax.set_axis_off()

    buffer = StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)

    svg = buffer.getvalue()
    svg_start = svg.find("<svg")
    if svg_start >= 0:
        svg = svg[svg_start:]

    tree_data = {
        "svg": svg,
        "width": 1400,
        "height": 800,
        "nodes": [],
        "edges": [],
        "note": "Drzewo zostało wytrenowane na wybranych danych treningowych. Klikanie w poszczególne węzły nie jest w tej wersji aktywne.",
        "truncated": False
    }

    return model, tree_data


def build_tree_for_template(
    X,
    y,
    feature_names,
    class_names,
    test_size=0.2,
    max_depth=5,
    criterion="gini",
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    model, tree_data = train_tree_for_template(
        X_train,
        y_train,
        feature_names,
        class_names,
        max_depth=max_depth,
        criterion=criterion,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )

    return {
        "model": model,
        "tree": tree_data,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
