
import pandas as pd
import os, sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

sys.path.append(str(base_dir))

from src.data.data_preprocessing import preprocess_data

RAW = base_dir / "data" / "raw" / "telco_customer_churn.xlsx"
OUT = base_dir / "data" / "processed" / "telco_churn_processed.xlsx"

df = pd.read_excel(RAW)

df_processed = preprocess_data(df, target_col="Churn")

OUT.parent.mkdir(parents=True, exist_ok=True)

df_processed.to_excel(OUT, index=False)

