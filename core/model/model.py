class BaseTreeModel:
    key = ''
    name = ''
    description = ''
    estimator_name = ''
    library = ''
    algorithm_title = ''
    algorithm_note = ''
    parameters = ()

    def __init__(self, params=None, random_state=42):
        if params is None:
            params = self.default_params()
        self.params = params
        self.random_state = random_state
        self.estimator = None
        self.feature_names = []
        self.class_names = []

    @classmethod
    def default_params(cls):
        values = {}
        for spec in cls.parameters:
            values[spec['name']] = spec['default']
        return values

    def build_estimator(self):
        raise NotImplementedError

    def build_nodes(self):
        raise NotImplementedError

    def structure_info(self):
        raise NotImplementedError

    def fit(self, X, y, feature_names, class_names):
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)
        self.estimator = self.build_estimator()
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

    def feature_importances(self):
        return list(self.estimator.feature_importances_)

    def export_nodes(self):
        nodes = self.build_nodes()
        add_paths(nodes, 0)
        return nodes


def add_paths(nodes, root):
    stack = [(root, [])]
    while len(stack) > 0:
        node_id, path = stack.pop()
        node = nodes[node_id]
        if len(path) == 0:
            node['hover_lines'].append('Ścieżka od korzenia: brak (to korzeń drzewa)')
        else:
            node['hover_lines'].append('Ścieżka od korzenia:')
            for condition in path:
                node['hover_lines'].append('- ' + condition)
        for child in node['children']:
            stack.append((child['target'], path + [child['condition']]))
