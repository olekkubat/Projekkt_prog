import pandas as pd
import xgboost as xgb
import joblib
import os

def train():
    file_name = 'ufc_gold_dataset_final.csv'
    if not os.path.exists(file_name):
        print(f"BŁĄD: Nie znaleziono pliku {file_name}")
        return

    df = pd.read_csv(file_name)

    features = [
        'F1_Sig_Landed', 'F2_Sig_Landed',
        'F1_TD_Landed', 'F2_TD_Landed',
        'F1_KD', 'F2_KD'
    ]

    existing_features = [f for f in features if f in df.columns]
    
    if len(existing_features) == 0:
        print("BŁĄD: Nie znaleziono kluczowych statystyk w pliku!")
        return

    df = df.dropna(subset=existing_features + ['Winner'])
    X = df[existing_features]
    
    y = (df['Winner'] == df['Fighter_1']).astype(int)

    print(f"Trenowanie na kolumnach: {existing_features}")
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X, y)

    joblib.dump(model, 'ufc_model_pro.pkl')
    print("Model ufc_model_pro.pkl został wygenerowany.")
    print(f"Skuteczność na danych: {model.score(X, y)*100:.2f}%")

if __name__ == "__main__":
    train()