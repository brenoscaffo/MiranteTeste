import lightgbm as lgbm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import sklearn
import mlflow

def tune_hiperparametros(X, y):
    """
    Otimização de hiperparâmetros do modelo LightGBM

    Argumentos:
        X (DataFrame) = features
        y (Series )= target
    """

    modelo = {
    "Light": {
        "model": lgbm.LGBMClassifier(),
        "params": {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [300, 500,900],
            'max_depth': [3,5,6,8],
            'subsample': [0.5, 0.75, 1]
            }
        }
    }

    for nome, info in modelo.items():
        with mlflow.start_run(run_name=nome):

            grid = GridSearchCV(
                estimator=info['model'],
                param_grid=info['params'],
                cv=3,
                verbose=4,
                scoring="roc_auc"
            )

            mlflow.lightgbm.autolog()

            grid.fit(X, y)

            #melhor modelo
            best_model = grid.best_estimator_
            # melhor modelo
            pred = best_model.predict(X)
            probabilidade_teste = best_model.predict_proba(X)[:,1]
            
            
            # =======================
            # Predições teste
            # =======================

            y_test_pred = (probabilidade_teste >= 0.3850211014145544).astype(int)

            precisao_teste = precision_score(y, y_test_pred)
            recall_teste = recall_score(y, y_test_pred)
            f1_teste = f1_score(y, y_test_pred)
            roc_auc_teste = roc_auc_score(y, probabilidade_teste)
            # registro de parametros
            mlflow.log_params(grid.best_params_)

            # registro de metricas
            mlflow.log_metrics({
                "Precisao_teste": precisao_teste,
                "Recall_teste": recall_teste,
                "F1_teste": f1_teste,
                "Curva_roc_teste": roc_auc_teste
            })

            print(f'{nome} finalizado!')
