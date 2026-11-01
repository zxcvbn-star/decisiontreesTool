from sklearn.tree import DecisionTreeClassifier

from .model import BaseTreeModel


class SklearnDecisionTree(BaseTreeModel):
    key = 'decision_tree'
    name = 'Drzewo decyzyjne (CART)'
    description = (
        'Pojedyncze drzewo budowane algorytmem CART. Każdy węzeł zadaje pytanie o jedną cechę '
        'i dzieli dane na dwie grupy, a liście podają przewidywaną klasę.'
    )
    estimator_name = 'DecisionTreeClassifier'
    library = 'scikit-learn'
    algorithm_title = 'Opis algorytmu CART'
    algorithm_note = (
        "Algorytm CART w każdym węźle wybiera cechę i próg, które najlepiej rozdzielają klasy, czyli dają "
        "najmniejszą nieczystość (na przykład Gini). Powtarza to, aż osiągnie maksymalną głębokość albo węzły "
        "zawierające próbki jednej klasy. Dokładność treningowa wyraźnie wyższa od testowej oznaczałaby przeuczenie."
    )
    parameters = (
        {
            'name': 'criterion',
            'label': 'Kryterium podziału',
            'kind': 'choice',
            'default': 'gini',
            'choices': (('gini', 'Gini'), ('entropy', 'Entropia'), ('log_loss', 'Log loss')),
            'help': 'Miara, według której drzewo ocenia jakość podziału węzła.',
        },
        {
            'name': 'max_depth',
            'label': 'Maksymalna głębokość',
            'kind': 'int',
            'min': 1,
            'max': 10,
            'default': 3,
            'help': 'Liczba poziomów drzewa (1 do 10). Większa wartość zwiększa ryzyko przeuczenia.',
        },
        {
            'name': 'min_samples_split',
            'label': 'Minimum próbek do podziału',
            'kind': 'int',
            'min': 2,
            'max': 40,
            'default': 2,
            'help': 'Węzeł z mniejszą liczbą próbek nie będzie już dzielony (2 do 40).',
        },
        {
            'name': 'min_samples_leaf',
            'label': 'Minimum próbek w liściu',
            'kind': 'int',
            'min': 1,
            'max': 20,
            'default': 1,
            'help': 'Najmniejsza dozwolona liczba próbek w liściu (1 do 20).',
        },
    )

    def build_estimator(self):
        return DecisionTreeClassifier(
            criterion=self.params['criterion'],
            max_depth=self.params['max_depth'],
            min_samples_split=self.params['min_samples_split'],
            min_samples_leaf=self.params['min_samples_leaf'],
            random_state=self.random_state,
        )

    def structure_info(self):
        return {
            'nodes': self.estimator.tree_.node_count,
            'leaves': self.estimator.get_n_leaves(),
            'depth': self.estimator.get_depth(),
        }

    def build_nodes(self):
        tree = self.estimator.tree_
        nodes = {}
        for node_id in range(tree.node_count):
            nodes[node_id] = self.describe_node(tree, node_id)
        return nodes

    def describe_node(self, tree, node_id):
        samples = int(tree.n_node_samples[node_id])
        impurity = round(float(tree.impurity[node_id]), 3)
        criterion = self.params['criterion']

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

        parts = []

        for index in range(len(shares)):
            parts.append(self.class_names[index] + ' ' + str(round(shares[index] * weighted)))

        distribution = ', '.join(parts)
        majority = self.class_names[best_class]

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
                    f'Nieczystość ({criterion}): {impurity}',
                ],
                'children': [],
            }

        feature = self.feature_names[int(tree.feature[node_id])]
        threshold = f'{float(tree.threshold[node_id]):.4}'

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
                f'Nieczystość ({criterion}): {impurity}',
            ],
            'children': [
                {'target': int(tree.children_left[node_id]), 'label': 'tak', 'condition': f'{feature} <= {threshold}'},
                {'target': int(tree.children_right[node_id]), 'label': 'nie', 'condition': f'{feature} > {threshold}'},
            ],
        }
