from sklearn.ensemble import RandomForestClassifier

from ..model import BaseTreeModel
from ..sklearn_nodes import build_nodes_from_estimator, tree_info


class SklearnRandomForest(BaseTreeModel):
    key = 'random_forest'
    name = 'Las losowy (Random Forest)'
    description = (
        'Zespół wielu drzew uczonych na losowych próbkach danych. '
        'Wynik to głosowanie wszystkich drzew. Na wykresie widać jedno wybrane drzewo lasu.'
    )
    estimator_name = 'RandomForestClassifier'
    library = 'scikit-learn'
    algorithm_title = 'Opis algoryatmu Random Forest'
    algorithm_note = (
        "Las losowy uczy wiele drzew CART. Każde drzewo widzi inną, losową próbkę danych (bootstrap), a przy każdym "
        "podziale rozważa tylko losowy podzbiór cech. Ostateczną klasę wybierają wszystkie drzewa przez głosowanie, "
        "co zmniejsza ryzyko przeuczenia pojedynczego drzewa. Na wykresie pokazane jest jedno, wybrane suwakiem drzewo lasu."
    )
    parameters = (
        {
            'name': 'n_estimators',
            'label': 'Liczba drzew w lesie',
            'kind': 'int',
            'min': 1,
            'max': 200,
            'default': 30,
            'help': 'Ile drzew tworzy las (1 do 200).',
        },
        {
            'name': 'criterion',
            'label': 'Kryterium podziału',
            'kind': 'choice',
            'default': 'gini',
            'choices': (('gini', 'Gini'), ('entropy', 'Entropia'), ('log_loss', 'Log loss')),
            'help': 'Miara jakości podziału w każdym drzewie.',
        },
        {
            'name': 'max_depth',
            'label': 'Maksymalna głębokość',
            'kind': 'int',
            'min': 1,
            'max': 10,
            'default': 3,
            'help': 'Liczba poziomów każdego drzewa (1 do 10).',
        },
        {
            'name': 'min_samples_split',
            'label': 'Minimum próbek do podziału',
            'kind': 'int',
            'min': 2,
            'max': 40,
            'default': 2,
            'help': 'Węzeł z mniejszą liczbą próbek nie będzie dzielony (2 do 40).',
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
        {
            'name': 'tree_index',
            'label': 'Wyświetlane drzewo (numer)',
            'kind': 'int',
            'min': 1,
            'max': 200,
            'default': 1,
            'help': 'Które drzewo lasu pokazać na wykresie (1 do 200, ale nie więcej niż liczba drzew).',
        },
    )

    def build_estimator(self):
        return RandomForestClassifier(
            n_estimators=self.params['n_estimators'],
            criterion=self.params['criterion'],
            max_depth=self.params['max_depth'],
            min_samples_split=self.params['min_samples_split'],
            min_samples_leaf=self.params['min_samples_leaf'],
            random_state=self.random_state,
        )

    def displayed(self):
        total = len(self.estimator.estimators_)
        index = self.clamp_index(total)
        return self.estimator.estimators_[index - 1], index, total

    def structure_info(self):
        member, index, total = self.displayed()
        info = tree_info(member)
        info['caption'] = f'Wyświetlane drzewo {index} z {total} w lesie.'
        info['extras'] = [('Liczba drzew w lesie', total), ('Wyświetlane drzewo', f'{index} z {total}')]
        return info

    def build_nodes(self):
        member, index, total = self.displayed()
        return build_nodes_from_estimator(
            member,
            self.feature_names,
            self.class_names,
            self.params['criterion'],
            [f'Drzewo lasu: {index} z {total}'],
        )
