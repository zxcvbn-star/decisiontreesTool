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


def describe_node(model, feature_names, class_names, node_id):
    tree = model.tree_
    samples = int(tree.n_node_samples[node_id])
    impurity = round(float(tree.impurity[node_id]), 4)
    criterion = model.criterion

    if tree.children_left[node_id] == -1:
        values = tree.value[node_id][0]
        total = 0.0

        for value in values:
            total = total + float(value)

        shares = []

        for value in values:
            if total > 0:
                shares.append(float(value) / total)
            else:
                shares.append(float(value))

        weighted = float(tree.weighted_n_node_samples[node_id])
        best_class = 0

        for index in range(len(shares)):
            if shares[index] > shares[best_class]:
                best_class = index
        label = class_names[int(model.classes_[best_class])]

        parts = []

        for index in range(len(shares)):
            name = class_names[int(model.classes_[index])]
            parts.append(name + ': ' + str(round(shares[index] * weighted)))

        distribution = ', '.join(parts)
        hover = (
            '<b>Liść</b><br>'
            f'Przewidywana klasa: {label}<br>'
            f'Próbek treningowych: {samples}<br>'
            f'Rozkład klas: {distribution}<br>'
            f'Nieczystość ({criterion}): {impurity}'
        )

        return {'leaf': True, 'title': str(label), 'subtitle': f'n={samples}', 'hover': hover}

    feature = feature_names[int(tree.feature[node_id])]
    threshold = f'{float(tree.threshold[node_id]):.4}'

    hover = (
        '<b>Węzeł decyzyjny</b><br>'
        f'Pytanie: czy {feature} &lt;= {threshold}?<br>'
        'Tak: w lewo, nie: w prawo<br>'
        f'Próbek w węźle: {samples}<br>'
        f'Nieczystość ({criterion}): {impurity}'
    )

    return {'leaf': False, 'title': feature, 'subtitle': f'&lt;= {threshold}', 'hover': hover}


def place(tree, node_id, depth, state, positions):
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]

    if left == -1:
        x = state['slot'] * (2.0 + 0.25)
        state['slot'] = state['slot'] + 1
    else:
        place(tree, left, depth + 1, state, positions)
        place(tree, right, depth + 1, state, positions)
        x = (positions[left]['x'] + positions[right]['x']) / 2.0
    positions[node_id] = {'x': x, 'depth': depth}

    if depth > state['depth']:
        state['depth'] = depth


def top_y(depth):
    return -depth * (0.62 + 0.8)


def build_tree_html(model, feature_names, class_names):
    tree = model.tree_
    infos = []

    for node_id in range(tree.node_count):
        infos.append(describe_node(model, feature_names, class_names, node_id))

    state = {'slot': 0, 'depth': 0}
    positions = {}
    place(tree, 0, 0, state, positions)
    total_width = state['slot'] * (2.0 + 0.25) - 0.25
    total_height = (state['depth'] + 1) * 0.62 + state['depth'] * 0.8

    edge_x = []
    edge_y = []
    label_x = []
    label_y = []
    label_text = []

    for node_id in range(tree.node_count):
        if infos[node_id]['leaf']:
            continue
        parent_x = positions[node_id]['x'] + 2.0 / 2.0
        parent_y = top_y(positions[node_id]['depth']) - 0.62
        for child_id in (tree.children_left[node_id], tree.children_right[node_id]):
            child_x = positions[child_id]['x'] + 2.0 / 2.0
            child_y = top_y(positions[child_id]['depth'])
            edge_x.extend([parent_x, child_x, None])
            edge_y.extend([parent_y, child_y, None])
            label_x.append((parent_x + child_x) / 2.0)
            label_y.append((parent_y + child_y) / 2.0)
            if child_id == tree.children_left[node_id]:
                label_text.append('tak')
            else:
                label_text.append('nie')

    figure = go.Figure()

    figure.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line={'color': '#475569', 'width': 1.5},
        hoverinfo='skip',
        showlegend=False,
    ))

    center_x = []
    center_y = []
    node_texts = []

    for node_id in range(tree.node_count):
        info = infos[node_id]
        x0 = positions[node_id]['x']
        x1 = x0 + 2.0
        y1 = top_y(positions[node_id]['depth'])
        y0 = y1 - 0.62

        if info['leaf']:
            fill = '#d1e7dd'
            border = '#1b6b45'
            border_width = 3
            node_texts.append('<b>' + info['title'] + '</b><br>' + info['subtitle'])
        else:
            fill = '#cfe2ff'
            border = '#1f4e9c'
            border_width = 1.5
            node_texts.append(info['title'] + '<br>' + info['subtitle'])

        center_x.append((x0 + x1) / 2.0)
        center_y.append((y0 + y1) / 2.0)

        figure.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[y0, y0, y1, y1, y0],
            mode='lines',
            fill='toself',
            fillcolor=fill,
            line={'color': border, 'width': border_width},
            hoveron='points',
            hoverinfo='text',
            hovertext=info['hover'],
            showlegend=False,
        ))

    figure.add_trace(go.Scatter(
        x=center_x,
        y=center_y,
        mode='text',
        text=node_texts,
        textposition='middle center',
        textfont={'size': 12, 'color': '#111827'},
        hoverinfo='skip',
        showlegend=False,
    ))

    figure.add_trace(go.Scatter(
        x=label_x,
        y=label_y,
        mode='text',
        text=label_text,
        textposition='middle center',
        textfont={'size': 12, 'color': '#334155'},
        hoverinfo='skip',
        showlegend=False,
    ))

    figure.update_layout(
        width=round((total_width + 0.6) * 80),
        height=round((total_height + 0.6) * 100),
        margin={'l': 5, 'r': 5, 't': 5, 'b': 5},
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis={'visible': False, 'range': [-0.3, total_width + 0.3]},
        yaxis={'visible': False, 'range': [-total_height - 0.3, 0.3]},
        hoverlabel={'align': 'left'},
    )
    return figure.to_html(full_html=False, include_plotlyjs=True, config={'displayModeBar': False})


def build_context():
    model, iris, X_train, X_test, y_train, y_test = train_model()
    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)

    train_labels = list(y_train)
    train_counts = []

    for index in range(len(class_names)):
        train_counts.append(f'{class_names[index]}: {train_labels.count(index)}')

    data_rows = [
        {
            'label': 'Zbiór danych',
            'value': 'Iris (scikit-learn)'
        },
        {
            'label': 'Liczba próbek',
            'value': str(len(iris.target))
        },
        {
            'label': 'Cechy',
            'value': ', '.join(feature_names)
        },
        {
            'label': 'Klasy',
            'value': ', '.join(class_names)
        },
        {
            'label': 'Podział danych',
            'value': 'trening 80%, test 20%'
        },
        {
            'label': 'Próbki treningowe',
            'value': str(len(y_train))
        },
        {
            'label': 'Próbki testowe',
            'value': str(len(y_test))
        },
        {
            'label': 'Klasy w danych treningowych',
            'value': ', '.join(train_counts)
        },
    ]

    param_rows = [
        {
            'label': 'Algorytm',
            'value': 'CART (DecisionTreeClassifier)'
        },
        {
            'label': 'criterion (kryterium podziału)',
            'value': str(model.criterion)
        },
        {
            'label': 'max_depth (maksymalna głębokość)',
            'value': str(model.max_depth)
        },
        {
            'label': 'min_samples_split',
            'value': str(model.min_samples_split)
        },
        {
            'label': 'min_samples_leaf',
            'value': str(model.min_samples_leaf)
        },
        {
            'label': 'random_state',
            'value': str(model.random_state)
        },
    ]

    train_accuracy = model.score(X_train, y_train) * 100
    test_accuracy = model.score(X_test, y_test) * 100

    result_rows = [
        {
            'label': 'Dokładność na danych treningowych',
            'value': f'{train_accuracy:.2f}%'
        },
        {
            'label': 'Dokładność na danych testowych',
            'value': f'{test_accuracy:.2f}%'
        },
        {
            'label': 'Liczba węzłów',
            'value': str(model.tree_.node_count)
        },
        {
            'label': 'Liczba liści',
            'value': str(model.get_n_leaves())
        },
        {
            'label': 'Głębokość drzewa',
            'value': str(model.get_depth())
        },
    ]

    pairs = []

    for index in range(len(feature_names)):
        pairs.append((float(model.feature_importances_[index]), feature_names[index]))

    pairs.sort(reverse=True)
    importances = []

    for value, name in pairs:
        percent = value * 100
        importances.append({'name': name, 'percent': f'{percent:.1f}', 'width': f'{percent:.1f}'})

    return {
        'tree_html': build_tree_html(model, feature_names, class_names),
        'data_rows': data_rows,
        'param_rows': param_rows,
        'result_rows': result_rows,
        'importances': importances,
    }


def home(request):
    return render(request, 'home.html', build_context())
