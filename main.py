import discord
from discord.ext import commands
import sqlite3
import joblib
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN') 

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'ufc_model_pro.pkl')
db_path = os.path.join(base_path, 'ufc_data.db')

model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Model UFC ULTRA-PRO (18 cech) załadowany pomyślnie.")
else:
    print("BŁĄD: Nie znaleziono pliku ufc_model_pro.pkl! Uruchom train_model.py")

def get_fighter_from_sql(name):
    if not os.path.exists(db_path): 
        return None
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT sig_landed, td_landed, age, wins, losses, height, reach, str_def, td_def 
        FROM fighters 
        WHERE LOWER(fighter_name) = LOWER(?)
    """, (name,))
    result = cursor.fetchone()
    
    conn.close()
    return result

@bot.event
async def on_ready():
    print(f'UFC_TYPER jest online jako {bot.user}')


@bot.command()
async def analyze(ctx, f1_name: str, f2_name: str):
    """symulacja walki"""
    if model is None: 
        await ctx.send("⚠️ Brak modelu XGBoost. Nie można wykonać analizy.")
        return

    f1 = get_fighter_from_sql(f1_name)
    f2 = get_fighter_from_sql(f2_name)

    if not f1 or not f2:
        missing = f1_name if not f1 else f2_name
        await ctx.send(f"❌ Nie znalazłem zawodnika `{missing}` w bazie danych SQL.")
        return

    f1_sig, f1_td, f1_age, f1_w, f1_l, f1_h, f1_r, f1_sd, f1_td_def = f1
    f2_sig, f2_td, f2_age, f2_w, f2_l, f2_h, f2_r, f2_sd, f2_td_def = f2

    feat_1 = np.array([[
        f1_sig, f2_sig, f1_td, f2_td, f1_age, f2_age, f1_w, f2_w, f1_l, f2_l,
        f1_h, f2_h, f1_r, f2_r, f1_sd, f2_sd, f1_td_def, f2_td_def
    ]], dtype=np.float32)
    probs_1 = model.predict_proba(feat_1)[0]
    
    feat_2 = np.array([[
        f2_sig, f1_sig, f2_td, f1_td, f2_age, f1_age, f2_w, f1_w, f2_l, f1_l,
        f2_h, f1_h, f2_r, f1_r, f2_sd, f1_sd, f2_td_def, f1_td_def
    ]], dtype=np.float32)
    probs_2 = model.predict_proba(feat_2)[0]

    final_p1 = ((probs_1[1] + probs_2[0]) / 2) * 100
    final_p2 = ((probs_1[0] + probs_2[1]) / 2) * 100

    color = discord.Color.red() if final_p1 > final_p2 else discord.Color.blue()
    embed = discord.Embed(title="🥊 Symulacja UFC AI Engine [Ultra-Pro]", color=color)
    
    embed.add_field(
        name=f"🟥 {f1_name} ({int(f1_age)} lat)", 
        value=f"Szanse: **{final_p1:.1f}%**\nRekord: {int(f1_w)}-{int(f1_l)}\nZasięg: {f1_r}\"\nObrona stójka: {f1_sd}%\nObrona obalenia: {f1_td_def}%", 
        inline=True
    )
    embed.add_field(
        name=f"🟦 {f2_name} ({int(f2_age)} lat)", 
        value=f"Szanse: **{final_p2:.1f}%**\nRekord: {int(f2_w)}-{int(f2_l)}\nZasięg: {f2_r}\"\nObrona stójka: {f2_sd}%\nObrona obalenia: {f2_td_def}%", 
        inline=True
    )
    embed.set_footer(text="Model 18-cechowy analizuje: Warunki, Stójkę, Zapasy, Wiek oraz Rekordy.")
    
    await ctx.send(embed=embed)

@bot.command()
async def stats(ctx, *, name: str):
    data = get_fighter_from_sql(name)
    if not data:
        await ctx.send(f"❌ Brak zawodnika `{name}` w bazie danych SQL.")
        return
    
    sig, td, age, w, l, h, r, sd, tdd = data
    embed = discord.Embed(title=f"📊 Karta zawodnika: {name}", color=discord.Color.gold())
    embed.add_field(name="Metryka & Rekord", value=f"Wiek: `{int(age)} lat`\nRekord: `{int(w)}-{int(l)}`", inline=False)
    embed.add_field(name="Warunki fizyczne", value=f"Wzrost: `{h}`\nZasięg: `{r}\"`", inline=True)
    embed.add_field(name="Ofensywa", value=f"Ciosy/min: `{sig}`\nŚr. obaleń: `{td}`", inline=True)
    embed.add_field(name="Defensywa", value=f"Uniki stójka: `{sd}%`\nObrona obalenia: `{tdd}%`", inline=True)
    
    await ctx.send(embed=embed)

@bot.command()
async def pomoc(ctx):
    """Wyświetla listę komend."""
    msg = (
        "**🤖 Panel sterowania UFC_TYPER Pro:**\n"
        "1️⃣ `!analyze \"Zawodnik A\" \"Zawodnik B\"` - Generuje zaawansowaną analizę 18-cechową.\n"
        "2️⃣ `!stats \"Imię Nazwisko\"` - Wyciąga pełną kartę statystyk zawodnika z bazy SQL.\n"
        "3️⃣ `!pomoc` - Wyświetla menu.\n\n"
    )
    await ctx.send(msg)

if TOKEN is None:
    print("Brak zmiennej DISCORD_TOKEN w pliku .env")
else:
    bot.run(TOKEN)