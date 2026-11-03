from django import forms

from .model import DEFAULT_MODEL, MODEL_CLASSES, MODELS

INT_MESSAGES = {
    'required': 'To pole jest wymagane.',
    'invalid': 'Wpisz liczbę całkowitą.',
    'min_value': 'Wartość nie może być mniejsza niż %(limit_value)s.',
    'max_value': 'Wartość nie może być większa niż %(limit_value)s.',
}
CHOICE_MESSAGES = {
    'invalid_choice': 'Wybierz jedną z dostępnych opcji.',
}

DEFAULT_TEST_SIZE = 20
DEFAULT_RANDOM_STATE = 42

MODEL_CHOICES = []
for model_class in MODEL_CLASSES:
    MODEL_CHOICES.append((model_class.key, model_class.name))


def field_name(model_key, param_name):
    return f'{model_key}__{param_name}'


def build_field(spec):
    if spec['kind'] == 'choice':
        return forms.ChoiceField(
            label=spec['label'],
            choices=spec['choices'],
            initial=spec['default'],
            required=False,
            help_text=spec['help'],
            error_messages=CHOICE_MESSAGES,
        )
    return forms.IntegerField(
        label=spec['label'],
        min_value=spec['min'],
        max_value=spec['max'],
        initial=spec['default'],
        required=False,
        help_text=spec['help'],
        error_messages=INT_MESSAGES,
    )


class TreeForm(forms.Form):
    model = forms.ChoiceField(
        label='Model',
        choices=MODEL_CHOICES,
        initial=DEFAULT_MODEL,
        error_messages=CHOICE_MESSAGES,
    )
    test_size = forms.IntegerField(
        label='Udział danych testowych (%)',
        min_value=10,
        max_value=50,
        initial=DEFAULT_TEST_SIZE,
        help_text='Część danych odłożona do sprawdzenia modelu (10 do 50).',
        error_messages=INT_MESSAGES,
    )
    random_state = forms.IntegerField(
        label='Ziarno losowości',
        min_value=0,
        max_value=99999,
        initial=DEFAULT_RANDOM_STATE,
        help_text='Ta sama wartość daje ten sam podział danych (0 do 99999).',
        error_messages=INT_MESSAGES,
    )

    def __init__(self, *args, **kwargs):
        kwargs['label_suffix'] = ''
        super().__init__(*args, **kwargs)
        for model_class in MODEL_CLASSES:
            for spec in model_class.parameters:
                self.fields[field_name(model_class.key, spec['name'])] = build_field(spec)

    def clean(self):
        cleaned = super().clean()
        selected = cleaned.get('model')
        for model_class in MODEL_CLASSES:
            for spec in model_class.parameters:
                name = field_name(model_class.key, spec['name'])
                if model_class.key == selected:
                    if cleaned.get(name) in (None, '') and name not in self.errors:
                        self.add_error(name, 'To pole jest wymagane.')
                else:
                    self.errors.pop(name, None)
        return cleaned

    def selected_model_key(self):
        value = self['model'].value()
        if value in MODELS:
            return value
        return DEFAULT_MODEL

    def model_sections(self):
        selected = self.selected_model_key()
        sections = []
        for model_class in MODEL_CLASSES:
            fields = []
            for spec in model_class.parameters:
                fields.append(self[field_name(model_class.key, spec['name'])])
            checked = ''
            if model_class.key == selected:
                checked = 'checked'
            sections.append({
                'key': model_class.key,
                'name': model_class.name,
                'description': model_class.description,
                'checked': checked,
                'fields': fields,
            })
        return sections

    def data_fields(self):
        return [self['test_size'], self['random_state']]

    def has_errors(self):
        return self.is_bound and not self.is_valid()

    def chosen_values(self):
        if self.is_bound and self.is_valid():
            model_key = self.cleaned_data['model']
            model_class = MODELS[model_key]
            params = {}
            for spec in model_class.parameters:
                params[spec['name']] = self.cleaned_data[field_name(model_key, spec['name'])]
            return {
                'model': model_key,
                'params': params,
                'test_size': self.cleaned_data['test_size'],
                'random_state': self.cleaned_data['random_state'],
            }
        return {
            'model': DEFAULT_MODEL,
            'params': MODELS[DEFAULT_MODEL].default_params(),
            'test_size': DEFAULT_TEST_SIZE,
            'random_state': DEFAULT_RANDOM_STATE,
        }
