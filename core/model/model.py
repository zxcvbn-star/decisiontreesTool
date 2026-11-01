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
