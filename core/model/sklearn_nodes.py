def class_name_of(estimator, class_names, index):
    return class_names[int(estimator.classes_[index])]


def split_children(tree, node_id, feature, threshold):
    return [
        {'target': int(tree.children_left[node_id]), 'label': 'tak', 'condition': f'{feature} <= {threshold}'},
        {'target': int(tree.children_right[node_id]), 'label': 'nie', 'condition': f'{feature} > {threshold}'},
    ]


def describe_class_node(estimator, tree, node_id, feature_names, class_names, impurity_name):
    samples = int(tree.n_node_samples[node_id])
    impurity = round(float(tree.impurity[node_id]), 3)
    values = tree.value[node_id][0]
    total = 0.0
    for value in values:
        total = total + float(value)
    weighted = float(tree.weighted_n_node_samples[node_id])
    shares = []
    for value in values:
        if total > 0:
            shares.append(float(value) / total)
        else:
            shares.append(float(value))
    best = 0
    for index in range(len(shares)):
        if shares[index] > shares[best]:
            best = index
    parts = []
    for index in range(len(shares)):
        name = class_name_of(estimator, class_names, index)
        parts.append(name + ' ' + str(round(shares[index] * weighted)))
    distribution = ', '.join(parts)
    majority = class_name_of(estimator, class_names, best)
    if tree.children_left[node_id] == -1:
        return {
            'leaf': True,
            'title': majority,
            'lines': [f'n = {samples}'],
            'hover_title': 'Liść',
            'hover_lines': [
                f'Przewidywana klasa: {majority}',
                f'Próbek treningowych: {samples}',
                f'Rozkład klas: {distribution}',
                f'Nieczystość ({impurity_name}): {impurity}',
            ],
            'children': [],
        }
    feature = feature_names[int(tree.feature[node_id])]
    threshold = f'{float(tree.threshold[node_id]):.4g}'
    return {
        'leaf': False,
        'title': feature,
        'lines': [f'<= {threshold}', f'n = {samples}'],
        'hover_title': 'Węzeł decyzyjny',
        'hover_lines': [
            f'Pytanie: czy {feature} <= {threshold}?',
            'Tak: próbka idzie w lewo, nie: w prawo',
            f'Próbek treningowych: {samples}',
            f'Rozkład klas: {distribution}',
            f'Dominująca klasa: {majority}',
            f'Nieczystość ({impurity_name}): {impurity}',
        ],
        'children': split_children(tree, node_id, feature, threshold),
    }


def describe_value_node(tree, node_id, feature_names, impurity_name):
    samples = int(tree.n_node_samples[node_id])
    impurity = round(float(tree.impurity[node_id]), 4)
    value = float(tree.value[node_id][0][0])
    if tree.children_left[node_id] == -1:
        return {
            'leaf': True,
            'title': f'{value:.3f}',
            'lines': [f'n = {samples}'],
            'hover_title': 'Liść',
            'hover_lines': [
                f'Wartość liścia: {value:.4f}',
                'Ta wartość jest dodawana do wyniku klasy w całym zespole',
                f'Próbek treningowych: {samples}',
                f'Błąd ({impurity_name}): {impurity}',
            ],
            'children': [],
        }
    feature = feature_names[int(tree.feature[node_id])]
    threshold = f'{float(tree.threshold[node_id]):.4g}'
    return {
        'leaf': False,
        'title': feature,
        'lines': [f'<= {threshold}', f'n = {samples}'],
        'hover_title': 'Węzeł decyzyjny',
        'hover_lines': [
            f'Pytanie: czy {feature} <= {threshold}?',
            'Tak: próbka idzie w lewo, nie: w prawo',
            f'Próbek treningowych: {samples}',
            f'Średnia wartość w węźle: {value:.4f}',
            f'Błąd ({impurity_name}): {impurity}',
        ],
        'children': split_children(tree, node_id, feature, threshold),
    }


def build_nodes_from_estimator(estimator, feature_names, class_names, impurity_name, context_lines=None):
    tree = estimator.tree_
    is_classifier = hasattr(estimator, 'classes_')
    nodes = {}
    for node_id in range(tree.node_count):
        if is_classifier:
            node = describe_class_node(estimator, tree, node_id, feature_names, class_names, impurity_name)
        else:
            node = describe_value_node(tree, node_id, feature_names, impurity_name)
        if context_lines:
            node['hover_lines'] = list(context_lines) + node['hover_lines']
        nodes[node_id] = node
    return nodes


def tree_info(estimator):
    return {
        'nodes': estimator.tree_.node_count,
        'leaves': estimator.get_n_leaves(),
        'depth': estimator.get_depth(),
    }
