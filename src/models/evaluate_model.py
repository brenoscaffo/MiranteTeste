import sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

def metricas_modelo(model, X_test, y_test):
    """
    Avaliar um modelo LightGBM no dataset de teste.

    Args:
        model: Modelo treinado.
        X_test: Features de teste.
        y_test: Rótulo de teste.
    """

    probabilidade = model.predict_probab(X_test)[:,1]
    preds = (probabilidade >= 0.3850211014145544).astype(int)

    print(f'Precisão: {precision_score(y_test, preds):.4f}')
    print(f'Recall: {recall_score(y_test, preds):.4f}')
    print(f'F1 Score: {f1_score(y_test, preds):.4f}')
    print(f'Curva ROC: {roc_auc_score(y_test, probabilidade):.4f}')