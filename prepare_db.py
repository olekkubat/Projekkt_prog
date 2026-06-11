import pandas as pd
import sqlite3
import os

def prepare_sql():
    csv_file = 'ufc_fighters_final.csv'
    if not os.path.exists(csv_file):
        print(f"❌ BŁĄD: Brak pliku {csv_file}")
        return

    fighters_df = pd.read_csv(csv_file)
    conn = sqlite3.connect('ufc_data.db')

    mapping = {
        'Fighter_Name': 'fighter_name',
        'SLpM': 'sig_landed', 
        'TD_Avg': 'td_landed',
        'Wins': 'wins',
        'Losses': 'losses',
        'Height': 'height',
        'Reach': 'reach',
        'Str_Def': 'str_def',
        'TD_Def': 'td_def'
    }
    
    existing_mapping = {k: v for k, v in mapping.items() if k in fighters_df.columns}
    df_clean = fighters_df[list(existing_mapping.keys())].copy()
    df_clean = df_clean.rename(columns=existing_mapping)
    
    if 'DOB' in fighters_df.columns:
        dob_parsed = pd.to_datetime(fighters_df['DOB'], errors='coerce')
        df_clean['age'] = 2026 - dob_parsed.dt.year
        df_clean['age'] = df_clean['age'].fillna(30).astype(int)
    else:
        df_clean['age'] = 30

    for col in ['str_def', 'td_def']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace('%', '', regex=False)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(50.0)

    if 'height' in df_clean.columns:
        df_clean['height'] = df_clean['height'].astype(str).str.replace('"', '', regex=False)
        df_clean['height'] = pd.to_numeric(df_clean['height'], errors='coerce').fillna(70.0)
    if 'reach' in df_clean.columns:
        df_clean['reach'] = df_clean['reach'].astype(str).str.replace('"', '', regex=False)
        df_clean['reach'] = pd.to_numeric(df_clean['reach'], errors='coerce').fillna(72.0)

    df_clean = df_clean.drop_duplicates(subset=['fighter_name'], keep='first')
    df_clean = df_clean.fillna(0)

    df_clean.to_sql('fighters', conn, if_exists='replace', index=False)
    
    conn.commit()
    conn.close()
    print(f"BAZA SQL ROZBUDOWANA  Kolumny: {list(df_clean.columns)}")

if __name__ == "__main__":
    prepare_sql()