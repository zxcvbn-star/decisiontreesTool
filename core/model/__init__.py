from .sklearnDecisionTree import SklearnDecisionTree

MODEL_CLASSES = (SklearnDecisionTree,)
MODELS = {}
for model_class in MODEL_CLASSES:
    MODELS[model_class.key] = model_class
DEFAULT_MODEL = MODEL_CLASSES[0].key
