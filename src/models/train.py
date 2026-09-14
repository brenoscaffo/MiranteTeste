# Importando as bibliotecas necessarias
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn import model_selection
from sklearn import pipeline
from sklearn import metrics
from sklearn import tree
from pathlib import Path
import lightgbm  as lgbm
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

# Importando a tabela processada
base_dir = base_dir = Path(__file__).resolve().parent.parent
OUT = base_dir / "data" / "processed" / "telco_churn_processed.xlsx"
df = pd.read_excel(OUT)
df.head()

target = 'Churn'
features = df.drop(columns='Churn', axis=1)

y = df[target]
X = df[features]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f'Base treino: {y_train.shape[0]} e Percetual de Churn {100*y_train.mean():.2f}%')
print(f'Base treino: {y_test.shape[0]} e Percetual de Churn {100*y_test.mean():.2f}%')

model = lgbm.LGBMClassifier()
threshold = 0.3850211014145544

with mlflow.start_run():
    model.fit(X_train, y_train)
    probabilidade = model.predict_proba(X_test)[:,1]
    predicao = (probabilidade >= threshold)
    prec = precision_score(y_test, predicao)
    rec = recall_score(y_test, predicao)
    f1 = f1_score(y_test, predicao)
    roc = roc_auc_score(y_test, probabilidade)

    # Log params, metricas e modelo
    mlflow.log_params(model.get_params())
    mlflow.log_metric("precisao", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1-score", f1)
    mlflow.log_metric("curva_roc", roc)
    mlflow.sklearn.log_model(model, "model")

    train_ds = mlflow.data.from_pandas(df, source="training_data")
    mlflow.log_input(train_ds, context="training")

    print(f"Model treinado. \nPrecisão: {prec:.4f} \nRecall: {rec:.4f} \nF1-Score: {f1:.4f} \nCurva ROC: {roc:.4f}")
    
