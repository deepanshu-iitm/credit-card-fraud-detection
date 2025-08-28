from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def make_preprocessor():
    scale_cols = ['Amount', 'Time']
    transformers = [
        ('scale', StandardScaler(), scale_cols),
        # pass-through for PCA features
    ]
    preprocessor = ColumnTransformer(
        transformers,
        remainder='passthrough'
    )
    return preprocessor

def make_pipeline(estimator, use_smote=True, random_state=42):
    steps = []
    steps.append(('pre', make_preprocessor()))
    if use_smote:
        steps.append(('smote', SMOTE(random_state=random_state)))
    steps.append(('clf', estimator))
    pipe = Pipeline(steps)
    return pipe
