import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    for coluna in ["customerID", "CustomerID", "customer_id"]:
        if coluna in df.columns:
            df = df.drop(columns=[coluna])

    # Normalizando as variaveis numericas não binarias
    colunas_numericas = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()

    df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])
    # Remapeando a target
    if target_col in df.columns and df[target_col].dtype == "object":
            df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    # Transformando as categoricas em dummies
    df = pd.get_dummies(df, columns=df.columns[df.dtypes == "object"], drop_first=True, dtype=int)

    df.columns = df.columns.str.replace(" ", "_") 

    return df 