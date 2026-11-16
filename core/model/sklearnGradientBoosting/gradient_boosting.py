from sklearn.ensemble import GradientBoostingClassifier

from ..model import BaseTreeModel
from ..sklearn_nodes import build_nodes_from_estimator, tree_info


class SklearnGradientBoosting(BaseTreeModel):
    key = 'gradient_boosting'
    name = 'Wzmacnianie gradientowe (Gradient Boosted Trees)'
    description = (
        'Zespół drzew budowanych kolejno, z których każde poprawia błędy poprzednich. '
        'Na wykresie widać jedno wybrane drzewo regresyjne zespołu.'
    )
    estimator_name = 'GradientBoostingClassifier'
    library = 'scikit-learn'
    algorithm_title = 'Opis algorytmu Gradient Boosted Trees'
    algorithm_note = (
        "Wzmacnianie gradientowe buduje drzewa kolejno: każde następne uczy się poprawiać błędy dotychczasowego zespołu. "
        "Dla każdej klasy w każdym etapie powstaje osobne, płytkie drzewo regresyjne, a ich wyniki są sumowane i przeliczane "
        "na prawdopodobieństwa. Liście pokazanego drzewa nie zawierają klas, tylko wartości, o które zmienia się wynik danej klasy. "
        "Współczynnik uczenia określa, jak mocno każde drzewo wpływa na wynik."
    )
    parameters = (
        {
            'name': 'n_estimators',
            'label': 'Liczba etapów boostingu',
            'kind': 'int',
            'min': 1,
            'max': 100,
            'default': 30,
            'help': 'Ile etapów ma zespół (1 do 100). W każdym etapie powstaje po jednym drzewie na klasę.',
        },
        {
            'name': 'learning_rate',
            'label': 'Współczynnik uczenia',
            'kind': 'float',
            'min': 0.01,
            'max': 1.0,
            'step': 0.01,
            'default': 0.1,
            'help': 'Jak mocno każde drzewo zmienia wynik (0.01 do 1).',
        },
        {
            'name': 'max_depth',
            'label': 'Maksymalna głębokość',
            'kind': 'int',
            'min': 1,
            'max': 6,
            'default': 3,
            'help': 'Liczba poziomów każdego drzewa (1 do 6).',
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
            'max': 300,
            'default': 1,
            'help': 'Które drzewo zespołu pokazać (1 do 300, ale nie więcej niż etapy razy klasy). Kolejność: etap po etapie, w każdym klasy po kolei.',
        },
    )

    def build_estimator(self):
        return GradientBoostingClassifier(
            n_estimators=self.params['n_estimators'],
            learning_rate=self.params['learning_rate'],
            max_depth=self.params['max_depth'],
            min_samples_leaf=self.params['min_samples_leaf'],
            random_state=self.random_state,
        )

    def displayed(self):
        stages = self.estimator.estimators_
        per_stage = stages.shape[1]
        total = stages.shape[0] * per_stage
        index = self.clamp_index(total)
        stage = (index - 1) // per_stage
        class_index = (index - 1) % per_stage
        if per_stage == 1:
            class_label = self.class_names[1]
        else:
            class_label = self.class_names[class_index]
        return stages[stage, class_index], index, total, stage + 1, class_label

    def structure_info(self):
        member, index, total, stage, class_label = self.displayed()
        info = tree_info(member)
        info['caption'] = f'Wyświetlane drzewo {index} z {total}: etap {stage}, klasa {class_label}.'
        info['extras'] = [
            ('Liczba drzew w zespole', total),
            ('Wyświetlane drzewo', f'{index} z {total}'),
            ('Etap i klasa drzewa', f'{stage}, {class_label}'),
        ]
        return info

    def build_nodes(self):
        member, index, total, stage, class_label = self.displayed()
        return build_nodes_from_estimator(
            member,
            self.feature_names,
            self.class_names,
            member.criterion,
            [f'Drzewo zespołu: {index} z {total} (etap {stage}, klasa {class_label})'],
        )
