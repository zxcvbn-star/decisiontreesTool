from sklearn.tree import DecisionTreeClassifier

from ..model import BaseTreeModel
from ..sklearn_nodes import build_nodes_from_estimator, tree_info


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
        info = tree_info(self.estimator)
        info['caption'] = 'Pojedyncze drzewo modelu.'
        info['extras'] = []
        return info

    def build_nodes(self):
        return build_nodes_from_estimator(
            self.estimator, self.feature_names, self.class_names, self.params['criterion']
        )
