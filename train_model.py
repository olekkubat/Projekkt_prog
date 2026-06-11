import pandas as pd
import xgboost as xgb
import sqlite3
import joblib
import os
import numpy as np

def train():
    csv_file = 'ufc_gold_dataset_final.csv'
    db_file = 'ufc_data.db'
    
    if not os.path.exists(csv_file) or not os.path.exists(db_file):
        print("❌ BŁĄD: Brak plików bazowych do treningu!")
        return

    df_fights = pd.read_csv(csv_file)
    conn = sqlite3.connect(db_file)
    df_fighters = pd.read_sql_query("SELECT * FROM fighters", conn)
    conn.close()

    df_fights['Fighter_1'] = df_fights['Fighter_1'].str.strip().str.lower()
    df_fights['Fighter_2'] = df_fights['Fighter_2'].str.strip().str.lower()
    df_fighters['fighter_name'] = df_fighters['fighter_name'].str.strip().str.lower()

    df_fights = df_fights.merge(df_fighters, left_on='Fighter_1', right_on='fighter_name', how='inner').rename(columns={
        'sig_landed': 'F1_Sig_Landed', 'td_landed': 'F1_TD_Landed', 'age': 'F1_Age',
        'wins': 'F1_Wins', 'losses': 'F1_Losses', 'height': 'F1_Height',
        'reach': 'F1_Reach', 'str_def': 'F1_Str_Def', 'td_def': 'F1_TD_Def'
    })
    if 'fighter_name' in df_fights.columns: df_fights = df_fights.drop(columns=['fighter_name'])

    df_fights = df_fights.merge(df_fighters, left_on='Fighter_2', right_on='fighter_name', how='inner').rename(columns={
        'sig_landed': 'F2_Sig_Landed', 'td_landed': 'F2_TD_Landed', 'age': 'F2_Age',
        'wins': 'F2_Wins', 'losses': 'F2_Losses', 'height': 'F2_Height',
        'reach': 'F2_Reach', 'str_def': 'F2_Str_Def', 'td_def': 'F2_TD_Def'
    })
    if 'fighter_name' in df_fights.columns: df_fights = df_fights.drop(columns=['fighter_name'])

    features = [
        'F1_Sig_Landed', 'F2_Sig_Landed',
        'F1_TD_Landed', 'F2_TD_Landed',
        'F1_Age', 'F2_Age',
        'F1_Wins', 'F2_Wins',
        'F1_Losses', 'F2_Losses',
        'F1_Height', 'F2_Height',
        'F1_Reach', 'F2_Reach',
        'F1_Str_Def', 'F2_Str_Def',
        'F1_TD_Def', 'F2_TD_Def'
    ]

    df_fights = df_fights.dropna(subset=features + ['Winner'])
    
    X_raw = df_fights[features].copy()
    X_raw = X_raw.loc[:, ~X_raw.columns.duplicated()].to_numpy(dtype=np.float32)
    
    winner_array = df_fights['Winner'].str.strip().str.lower().to_numpy()
    f1_array = df_fights['Fighter_1'].str.strip().str.lower().to_numpy()
    y_raw = (winner_array == f1_array).astype(np.int32)

    print(f"Macierz treningowa gotowa Kształt danych: {X_raw.shape} ")
    
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.06, random_state=42)
    model.fit(X_raw, y_raw)

    base_path = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(model, os.path.join(base_path, 'ufc_model_pro.pkl'))
    print("SUKCES Model UFC ) został pomyślnie wytrenowany i zapisany")

if __name__ == "__main__":
    train()