from .forms import TreeForm
from .model import MODELS
from .model.data import DATASET_NAME, load_split
from .visualization import build_tree_html


def make_rows(pairs):
    rows = []
    for label, value in pairs:
        rows.append({'label': label, 'value': str(value)})
    return rows


def param_display(spec, value):
    if spec['kind'] == 'choice':
        for choice_value, choice_label in spec['choices']:
            if choice_value == value:
                return choice_label
    return value


def build_context(query):
    if query:
        form = TreeForm(query)
    else:
        form = TreeForm()

    values = form.chosen_values()
    model_class = MODELS[values['model']]
    data = load_split(values['test_size'], values['random_state'])
    model = model_class(values['params'], values['random_state'])
    model.fit(data['X_train'], data['y_train'], data['feature_names'], data['class_names'])
    info = model.structure_info()
    train_accuracy = model.score(data['X_train'], data['y_train']) * 100
    test_accuracy = model.score(data['X_test'], data['y_test']) * 100
    train_count = len(data['y_train'])
    test_count = len(data['y_test'])
    train_percent = round(train_count / data['n_samples'] * 100)
    test_percent = round(test_count / data['n_samples'] * 100)

    data_rows = make_rows([
        ('Zbiór danych', DATASET_NAME),
        ('Liczba próbek', data['n_samples']),
        ('Liczba cech', len(data['feature_names'])),
        ('Cechy', ', '.join(data['feature_names'])),
        ('Klasy', ', '.join(data['class_names'])),
        ('Zbiór treningowy', f'{train_count} próbek ({train_percent}%)'),
        ('Zbiór testowy', f'{test_count} próbek ({test_percent}%)'),
    ])
    param_pairs = [('Model', model_class.name)]

    for spec in model_class.parameters:
        param_pairs.append((spec['name'], param_display(spec, values['params'][spec['name']])))
    param_pairs.append(('random_state', values['random_state']))
    result_rows = make_rows([
        ('Dokładność treningowa', f'{train_accuracy:.2f}%'),
        ('Dokładność testowa', f'{test_accuracy:.2f}%'),
        ('Liczba węzłów', info['nodes']),
        ('Liczba liści', info['leaves']),
        ('Głębokość drzewa', info['depth']),
    ])

    importances = []
    values_list = model.feature_importances()

    for index in range(len(data['feature_names'])):
        percent = f'{values_list[index] * 100:.1f}'
        importances.append(
            {
                'name': data['feature_names'][index],
                'percent': percent,
                'width': percent
            })

    return {
        'form': form,
        'has_errors': form.has_errors(),
        'model_sections': form.model_sections(),
        'data_fields': form.data_fields(),
        'tree_html': build_tree_html(model.export_nodes()),
        'data_rows': data_rows,
        'param_rows': make_rows(param_pairs),
        'result_rows': result_rows,
        'importances': importances,
        'algorithm_title': model_class.algorithm_title,
        'algorithm_note': model_class.algorithm_note,
    }
