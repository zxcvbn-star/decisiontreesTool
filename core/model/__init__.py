from core.model.sklearnDecisionStump.decision_stump import SklearnDecisionStump
from core.model.sklearnDecisionTree.sklearnDecisionTree import SklearnDecisionTree
from core.model.sklearnGradientBoosting.gradient_boosting import SklearnGradientBoosting
from core.model.sklearnRandomForest.randomForest import SklearnRandomForest

MODEL_CLASSES = (
    SklearnDecisionTree,
    SklearnDecisionStump,
    SklearnRandomForest,
    SklearnGradientBoosting,
)
MODELS = {}
for model_class in MODEL_CLASSES:
    MODELS[model_class.key] = model_class
DEFAULT_MODEL = MODEL_CLASSES[0].key

