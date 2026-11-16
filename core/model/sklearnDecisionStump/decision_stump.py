from sklearn.tree import DecisionTreeClassifier

from ..model import BaseTreeModel
from ..sklearn_nodes import build_nodes_from_estimator, tree_info


class SklearnDecisionStump(BaseTreeModel):
    key = 'decision_stump'
    name = 'Pniak decyzyjny (Decision Stump)'
    description = (
        'Najprostsze drzewo: jeden podział według jednej cechy i dwa liście. '
        'Bardzo słaby klasyfikator, ale podstawa metod zespołowych.'
    )
    estimator_name = 'DecisionTreeClassifier (max_depth = 1)'
    library = 'scikit-learn'
    algorithm_title = 'Opis pniaka decyzyjnego'
    algorithm_note = (
        "Pniak decyzyjny to drzewo o głębokości 1: algorytm wybiera jedną cechę i jeden próg, które najlepiej "
        "rozdzielają klasy, i na tym kończy. Sam jest bardzo słabym klasyfikatorem, szczególnie przy trzech klasach, "
        "ale jest podstawowym elementem metod zespołowych, takich jak AdaBoost."
    )
    parameters = (
        {
            'name': 'criterion',
            'label': 'Kryterium podziału',
            'kind': 'choice',
            'default': 'gini',
            'choices': (('gini', 'Gini'), ('entropy', 'Entropia'), ('log_loss', 'Log loss')),
            'help': 'Miara, według której pniak ocenia jakość podziału.',
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
    fixed_params = (('max_depth', 1),)

    def build_estimator(self):
        return DecisionTreeClassifier(
            criterion=self.params['criterion'],
            max_depth=1,
            min_samples_leaf=self.params['min_samples_leaf'],
            random_state=self.random_state,
        )

    def structure_info(self):
        info = tree_info(self.estimator)
        info['caption'] = 'Pniak: jedno drzewo o głębokości 1.'
        info['extras'] = []
        return info

    def build_nodes(self):
        return build_nodes_from_estimator(
            self.estimator, self.feature_names, self.class_names, self.params['criterion']
        )
