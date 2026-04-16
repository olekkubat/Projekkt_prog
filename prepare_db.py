import pandas as pd

def prepare():
    # Wczytujemy plik z zawodnikami
    fighters = pd.read_csv('ufc_fighters_final.csv')
    
    # Mapujemy Twoje specyficzne nazwy na standardowe dla bota
    # Zmień nazwy po lewej stronie, jeśli w Twoim CSV są inne!
    mapping = {
        'Fighter_Name': 'fighter_name',
        'F1_Sig_Landed': 'sig_landed', 
        'F1_TD_Landed': 'td_landed',
        'F1_KD': 'kd'
    }
    
    # Filtrujemy tylko te kolumny, które istnieją
    existing_mapping = {k: v for k, v in mapping.items() if k in fighters.columns}
    
    fighters_clean = fighters[existing_mapping.keys()].copy()
    fighters_clean = fighters_clean.rename(columns=existing_mapping)
    
    # Usuwamy duplikaty, zostawiając najnowsze statystyki
    fighters_clean = fighters_clean.drop_duplicates(subset=['fighter_name'], keep='first')
    
    fighters_clean.to_csv('fighters_db.csv', index=False)
    print(f"✅ Baza przygotowana. Kolumny: {fighters_clean.columns.tolist()}")

if __name__ == "__main__":
    prepare()