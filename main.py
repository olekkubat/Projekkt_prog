import discord
from discord.ext import commands
import pandas as pd
import joblib
import os
import numpy as np

TOKEN = 'MTQ5MjYzMzg1MDY2NzQ2Njc4Mg.GmIGon.xmNQZhxvb9VHeWdXlupOBYsNCryGlsZIbSXK0I' 

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'ufc_model_pro.pkl')
db_path = os.path.join(base_path, 'fighters_db.csv')


model = None
db = None

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model załadowany pomyślnie.")
else:
    print("BŁĄD: Nie znaleziono pliku ufc_model_pro.pkl! Uruchom train_model.py")

if os.path.exists(db_path):
    db = pd.read_csv(db_path)
    db.columns = [c.lower() for c in db.columns]
    print(f"Baza zawodników załadowana. Wykryte kolumny: {list(db.columns)}")
else:
    print("BŁĄD: Nie znaleziono pliku fighters_db.csv! Uruchom prepare_db.py")

@bot.event
async def on_ready():
    print(f'Bot UFC jest online jako {bot.user}')


@bot.command()
async def analyze(ctx, f1_name: str, f2_name: str):
    if model is None or db is None:
        await ctx.send("Bot nie jest w pełni skonfigurowany (brak modelu lub bazy).")
        return

    f1 = db[db['fighter_name'].str.lower() == f1_name.lower()]
    f2 = db[db['fighter_name'].str.lower() == f2_name.lower()]

    if f1.empty or f2.empty:
        missing = f1_name if f1.empty else f2_name
        await ctx.send(f"Nie znalazłem zawodnika `{missing}` w bazie danych.")
        return

    r1 = f1.iloc[0]
    r2 = f2.iloc[0]

    f1_sig = r1.get('sig_landed', 0)
    f2_sig = r2.get('sig_landed', 0)
    f1_td = r1.get('td_landed', 0)
    f2_td = r2.get('td_landed', 0)
    f1_kd = r1.get('kd', 0)
    f2_kd = r2.get('kd', 0)

    features = np.array([[f1_sig, f2_sig, f1_td, f2_td, f1_kd, f2_kd]])

    try:
        probs = model.predict_proba(features)[0]
        print(f"Dane wejściowe do modelu: {features}")
        p1 = probs[1] * 100
        p2 = probs[0] * 100

        embed = discord.Embed(
            title="Symulacja Walki UFC",
            description=f"Analiza starcia: **{f1_name}** vs **{f2_name}**",
            color=discord.Color.dark_red() if p1 > p2 else discord.Color.dark_blue()
        )
        embed.add_field(name=f"🟥 {f1_name}", value=f"Szanse: **{p1:.1f}%**", inline=True)
        embed.add_field(name=f"🟦 {f2_name}", value=f"Szanse: **{p2:.1f}%**", inline=True)
        
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Wystąpił błąd podczas analizy: {e}")



if TOKEN != 'MTQ5MjYzMzg1MDY2NzQ2Njc4Mg.GmIGon.xmNQZhxvb9VHeWdXlupOBYsNCryGlsZIbSXK0I':
    print("BLAD")
else:
    bot.run(TOKEN)