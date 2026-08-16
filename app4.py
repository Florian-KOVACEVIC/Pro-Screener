import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import requests
import io

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Pro Screener - Court Terme", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- TITRE & DESCRIPTION ---
st.title("Screener d'Opportunités Court Terme & Rebond")
st.markdown(
    "Analyse quotidienne automatisée des actions du S&P 500 en survente "
    "présentant un volume anormal et des signaux techniques de rebond."
)

# --- 1. RÉCUPÉRATION DES TICKERS DU S&P 500 ---
@st.cache_data(ttl=86400) # Mise en cache 24h
def load_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # En-tête pour simuler un vrai navigateur et éviter le blocage 403 / FileNotFoundError
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    response = requests.get(url, headers=headers)
    
    # pandas parse le contenu HTML directement
    table = pd.read_html(io.StringIO(response.text))
    df = table[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    return df[['Symbol', 'Security', 'GICS Sector']]

# --- 2. CALCUL DES INDICATEURS TECHNIQUES ---
def calculate_indicators(df_stock):
    # RSI (14 jours)
    delta = df_stock['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Bandes de Bollinger (20 jours, 2 stdev)
    sma20 = df_stock['Close'].rolling(20).mean()
    std20 = df_stock['Close'].rolling(20).std()
    bollinger_low = sma20 - (2 * std20)
    
    # Ratio de Volume (Volume jour / Moyenne 20j)
    vol_sma20 = df_stock['Volume'].rolling(20).mean()
    vol_ratio = df_stock['Volume'] / vol_sma20
    
    return rsi, bollinger_low, vol_ratio

# --- 3. ANALYSE ET SCRAPING DES DONNÉES ---
@st.cache_data(ttl=86400)
def fetch_and_analyze(df_sp500):
    tickers = df_sp500['Symbol'].tolist()
    # Téléchargement groupé des cours sur 90 jours
    data = yf.download(tickers, period="90d", interval="1d", group_by='ticker', progress=False)
    
    results = []
    
    for _, row in df_sp500.iterrows():
        symbol = row['Symbol']
        name = row['Security']
        sector = row['GICS Sector']
        
        try:
            df_s = data[symbol].dropna()
            if len(df_s) < 30: 
                continue
            
            close = df_s['Close']
            last_p = close.iloc[-1]
            prev_p = close.iloc[-2]
            var_day = ((last_p - prev_p) / prev_p) * 100
            var_5d = ((last_p - close.iloc[-5]) / close.iloc[-5]) * 100
            
            rsi_s, boll_low_s, vol_ratio_s = calculate_indicators(df_s)
            
            rsi = rsi_s.iloc[-1]
            boll_low = boll_low_s.iloc[-1]
            vol_ratio = vol_ratio_s.iloc[-1]
            
            # --- Algorithme du Score Opportunité (0 à 100) ---
            score = 0
            if rsi < 30: 
                score += 40
            elif rsi < 40: 
                score += 20
            
            if last_p <= boll_low: 
                score += 35
            if vol_ratio > 1.2: 
                score += 25  # Volume d'échange élevé accompagnant le mouvement
            
            results.append({
                'Ticker': symbol,
                'Nom': name,
                'Secteur': sector,
                'Prix ($)': round(last_p, 2),
                'Var. 1J (%)': round(var_day, 2),
                'Var. 5J (%)': round(var_5d, 2),
                'RSI (14)': round(rsi, 1) if not pd.isna(rsi) else 50,
                'Sous Bollinger': "Oui 🟢" if last_p <= boll_low else "Non 🔴",
                'Ratio Vol.': round(vol_ratio, 2) if not pd.isna(vol_ratio) else 1.0,
                'Score Opp.': score,
                'History': df_s # Conservé pour la génération du graphique
            })
        except Exception:
            continue
            
    return pd.DataFrame(results)

# --- CHARGEMENT ET TRAITEMENT DES DONNÉES ---
with st.spinner("Analyse du marché S&P 500 en cours..."):
    sp500_df = load_sp500()
    results_df = fetch_and_analyze(sp500_df)

# --- SIDEBAR (FILTRES D'ANALYSE) ---
st.sidebar.header("Filtres Stratégiques")

selected_sector = st.sidebar.selectbox("Secteur d'activité", ["Tous"] + list(results_df['Secteur'].unique()))
min_score = st.sidebar.slider("Score Opportunité Min.", 0, 100, 40, step=5)
rsi_max = st.sidebar.slider("RSI Max (Zone de survente)", 10, 50, 35)
volume_filter = st.sidebar.checkbox("Volume supérieur à la moyenne (> 1.0x)", value=True)

# Application des filtres utilisateur
filtered = results_df.copy()

if selected_sector != "Tous":
    filtered = filtered[filtered['Secteur'] == selected_sector]

filtered = filtered[
    (filtered['Score Opp.'] >= min_score) &
    (filtered['RSI (14)'] <= rsi_max)
]

if volume_filter:
    filtered = filtered[filtered['Ratio Vol.'] >= 1.0]

filtered = filtered.sort_values(by='Score Opp.', ascending=False)

# --- MÉTRIQUES GLOBALES EN HAUT DE PAGE ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Opportunités détectées", len(filtered))
col2.metric("RSI Moyen Sélection", round(filtered['RSI (14)'].mean(), 1) if len(filtered) > 0 else 0)
col3.metric("Baisse Moyenne 1J", f"{round(filtered['Var. 1J (%)'].mean(), 2)}%" if len(filtered) > 0 else "0%")
col4.metric("Dernière MAJ", datetime.date.today().strftime("%d/%m/%Y"))

# --- TABLEAU DE RÉSULTATS ---
st.subheader("Sélection des meilleures opportunités")

# Suppression de la colonne lourde 'History' pour l'affichage du tableau
display_df = filtered.drop(columns=['History'])

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Score Opp.": st.column_config.ProgressColumn(format="%d/100", min_value=0, max_value=100),
        "Prix ($)": st.column_config.NumberColumn(format="$ %.2f"),
        "Var. 1J (%)": st.column_config.NumberColumn(format="%.2f %%"),
        "Var. 5J (%)": st.column_config.NumberColumn(format="%.2f %%"),
        "Ratio Vol.": st.column_config.NumberColumn(format="%.2f x"),
    }
)

# Bouton d'exportation CSV
st.download_button(
    label="Exporter la sélection au format CSV",
    data=display_df.to_csv(index=False).encode('utf-8'),
    file_name=f'opportunites_bourse_{datetime.date.today()}.csv',
    mime='text/csv'
)

# --- SECTION GRAPHIQUE INTERACTIF ---
st.markdown("---")
st.subheader("Analyse Graphique Interactive")

if len(filtered) > 0:
    selected_ticker = st.selectbox("Sélectionner une action à analyser :", filtered['Ticker'].tolist())
    
    # Extraction de l'historique de l'action sélectionnée
    stock_data = filtered[filtered['Ticker'] == selected_ticker]['History'].values[0]
    
    # Graphique Plotly à sous-plots (Prix + RSI)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08, 
        row_heights=[0.7, 0.3]
    )
    
    # 1. Chandeliers japonais
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'], high=stock_data['High'],
        low=stock_data['Low'], close=stock_data['Close'],
        name="Prix"
    ), row=1, col=1)
    
    # 2. RSI
    rsi_vals, _, _ = calculate_indicators(stock_data)
    fig.add_trace(go.Scatter(
        x=stock_data.index, 
        y=rsi_vals, 
        name="RSI (14)", 
        line=dict(color='purple', width=1.5)
    ), row=2, col=1)
    
    # Lignes de seuils RSI (Survente 30 / Surachat 70)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    
    fig.update_layout(
        height=550, 
        xaxis_rangeslider_visible=False, 
        title=f"Évolution et Indicateurs - {selected_ticker}",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Aucune action ne correspond aux critères de recherche actuels.")