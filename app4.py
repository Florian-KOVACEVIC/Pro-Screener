"""
================================================================================
 PRO SCREENER — Opportunités Court Terme & Rebond (Multi-Marchés)
================================================================================
Application Streamlit pour détecter des actions/cryptos en survente présentant
un volume anormal et des signaux techniques de rebond, sur n'importe quel
marché : indices US (S&P 500, Nasdaq 100, Dow 30), indices européens
(CAC 40, DAX 40, FTSE 100), une sélection thématique "Tech & IA en vogue",
un panier de cryptomonnaies majeures, ou une liste de tickers personnalisée.

Architecture :
  1. Configuration & style (thème "terminal de trading")
  2. Registre des marchés (MarketConfig) + chargeurs d'univers
  3. Indicateurs techniques (RSI, Bollinger, Volume, MACD, range 52 sem.)
  4. Moteur de scoring d'opportunité
  5. Pipeline de récupération & d'analyse des données (mise en cache)
  6. Interface utilisateur (sidebar, KPIs, spotlight, onglets)
================================================================================
"""

import io
import datetime as dt
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION DE PAGE & STYLE
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Pro Screener — Multi-Marchés",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    """Injecte le thème visuel 'terminal de trading' (dark, dual-accent,
    typographie monospace pour les données chiffrées)."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        :root{
            --bg-void:#0A0E1A; --bg-panel:#111827; --bg-card:#161F32;
            --border:#232C42; --text-hi:#E7ECF5; --text-lo:#8492AC;
            --emerald:#22C55E; --emerald-soft:rgba(34,197,94,.14);
            --amber:#F5A623; --amber-soft:rgba(245,166,35,.14);
            --rose:#F0466E; --rose-soft:rgba(240,70,110,.14);
            --indigo:#6C8EFF;
        }

        html, body, [class*="css"]{ font-family:'Inter',sans-serif; }
        .stApp{ background:radial-gradient(ellipse at top, #0D1220 0%, var(--bg-void) 55%); }
        [data-testid="stHeader"]{ background:transparent; }

        h1,h2,h3,h4{ font-family:'Space Grotesk',sans-serif !important; color:var(--text-hi) !important; letter-spacing:-.01em; }
        p, span, label, div{ color:var(--text-hi); }
        .stCaption, [data-testid="stCaptionContainer"]{ color:var(--text-lo) !important; }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"]{ background:var(--bg-panel); border-right:1px solid var(--border); }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] textarea, [data-testid="stSidebar"] input{
            background:var(--bg-card) !important; color:var(--text-hi) !important; border-color:var(--border) !important;
        }

        /* ---------- Ticker tape (signature element) ---------- */
        .tape-wrap{ overflow:hidden; background:var(--bg-panel); border-top:1px solid var(--border);
                     border-bottom:1px solid var(--border); padding:9px 0; margin-bottom:22px; }
        .tape-track{ display:inline-flex; white-space:nowrap; animation:scroll-tape 45s linear infinite; }
        .tape-item{ font-family:'IBM Plex Mono',monospace; font-size:13px; padding:0 26px; color:var(--text-lo);
                     border-right:1px solid var(--border); }
        .tape-up{ color:var(--emerald); font-weight:600; } .tape-down{ color:var(--rose); font-weight:600; }
        @keyframes scroll-tape{ 0%{ transform:translateX(0);} 100%{ transform:translateX(-50%);} }
        @media (prefers-reduced-motion: reduce){ .tape-track{ animation:none; } }

        /* ---------- Hero ---------- */
        .hero{ padding:4px 0 18px 0; }
        .hero-badge{ display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:.08em;
                      color:var(--emerald); background:var(--emerald-soft); border:1px solid rgba(34,197,94,.35);
                      border-radius:999px; padding:4px 12px; margin-bottom:12px; text-transform:uppercase; }
        .hero h1{ font-size:2rem; margin:0 0 6px 0; }
        .hero p{ color:var(--text-lo); font-size:.98rem; max-width:760px; margin:0; }

        /* ---------- Metrics ---------- */
        [data-testid="stMetric"]{ background:var(--bg-card); border:1px solid var(--border); border-radius:14px;
                                    padding:14px 18px; }
        [data-testid="stMetricLabel"]{ color:var(--text-lo) !important; font-size:.8rem !important; }
        [data-testid="stMetricValue"]{ font-family:'IBM Plex Mono',monospace !important; color:var(--text-hi) !important; }

        /* ---------- Spotlight cards ---------- */
        .opp-card{ background:linear-gradient(160deg,var(--bg-card),var(--bg-panel)); border:1px solid var(--border);
                    border-radius:16px; padding:16px 18px; height:100%; }
        .opp-rank{ font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--text-lo); text-transform:uppercase; letter-spacing:.08em; }
        .opp-ticker{ font-family:'Space Grotesk',sans-serif; font-size:1.35rem; font-weight:700; color:var(--text-hi); margin:2px 0 0 0; }
        .opp-name{ color:var(--text-lo); font-size:.82rem; margin-bottom:10px; }
        .opp-price{ font-family:'IBM Plex Mono',monospace; font-size:1.05rem; color:var(--text-hi); }

        .badge{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:600;
                 font-family:'IBM Plex Mono',monospace; margin-right:6px; margin-top:6px; }
        .badge-buy{ background:var(--emerald-soft); color:var(--emerald); }
        .badge-warn{ background:var(--amber-soft); color:var(--amber); }
        .badge-neutral{ background:rgba(132,146,172,.14); color:var(--text-lo); }
        .badge-down{ background:var(--rose-soft); color:var(--rose); }

        /* ---------- Dataframe / tabs ---------- */
        [data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:12px; overflow:hidden; }
        .stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--border); }
        .stTabs [data-baseweb="tab"]{ font-family:'Space Grotesk',sans-serif; color:var(--text-lo); }
        .stTabs [aria-selected="true"]{ color:var(--emerald) !important; }

        .footnote{ color:var(--text-lo); font-size:.78rem; line-height:1.5; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_style()

# ══════════════════════════════════════════════════════════════════════════
# 2. REGISTRE DES MARCHÉS
# ══════════════════════════════════════════════════════════════════════════
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


@dataclass
class MarketConfig:
    key: str
    label: str
    loader: Optional[Callable[[], pd.DataFrame]]
    suffix: str = ""          # suffixe boursier requis par Yahoo Finance (ex: ".PA", ".DE", ".L")
    currency: str = "$"
    group_label: str = "Secteur"
    is_curated: bool = False  # True = liste maison, pas une composition officielle d'indice
    note: str = ""


def _best_wiki_table(url: str, ticker_keys: list[str], name_keys: list[str]) -> pd.DataFrame:
    """Récupère la page Wikipedia et retourne la première table qui contient
    à la fois une colonne 'ticker-like' et une colonne 'nom-like'. Robuste
    aux changements de mise en page (les tables de composants ne sont pas
    toujours à l'index 0)."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        has_ticker = any(any(k in c for k in ticker_keys) for c in cols)
        has_name = any(any(k in c for k in name_keys) for c in cols)
        if has_ticker and has_name:
            return t
    raise ValueError("table des composants introuvable sur la page Wikipedia")


def _col(df: pd.DataFrame, keys: list[str]) -> Optional[str]:
    for c in df.columns:
        if any(k in str(c).lower() for k in keys):
            return c
    return None


def _standardize(df: pd.DataFrame, ticker_keys, name_keys, sector_keys, clean_dot=True) -> pd.DataFrame:
    tcol, ncol, scol = _col(df, ticker_keys), _col(df, name_keys), _col(df, sector_keys)
    out = pd.DataFrame()
    out["Symbol"] = df[tcol].astype(str).str.strip()
    if clean_dot:
        out["Symbol"] = out["Symbol"].str.replace(".", "-", regex=False)  # ex: BRK.B -> BRK-B
    out["Nom"] = df[ncol].astype(str).str.strip() if ncol else out["Symbol"]
    out["Groupe"] = df[scol].astype(str).str.strip() if scol else "N/A"
    return out.drop_duplicates(subset="Symbol").reset_index(drop=True)


# ---- Chargeurs d'indices officiels (scraping Wikipedia en direct) ---------
# NB : par prudence, aucune liste de secours codée en dur n'est utilisée pour
# les indices officiels ci-dessous — une composition d'indice inventée ou
# obsolète serait trompeuse. En cas d'échec du scraping, l'app affiche une
# erreur claire plutôt que de substituer des données non fiables.

@st.cache_data(ttl=86400, show_spinner=False)
def load_sp500() -> pd.DataFrame:
    t = _best_wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["symbol"], ["security", "company"])
    return _standardize(t, ["symbol"], ["security", "company"], ["gics sector", "sector"])


@st.cache_data(ttl=86400, show_spinner=False)
def load_nasdaq100() -> pd.DataFrame:
    t = _best_wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100", ["ticker", "symbol"], ["company"])
    return _standardize(t, ["ticker", "symbol"], ["company"], ["gics sector", "sector"])


@st.cache_data(ttl=86400, show_spinner=False)
def load_dow30() -> pd.DataFrame:
    t = _best_wiki_table("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", ["symbol"], ["company"])
    return _standardize(t, ["symbol"], ["company"], ["industry"])


@st.cache_data(ttl=86400, show_spinner=False)
def load_cac40() -> pd.DataFrame:
    t = _best_wiki_table("https://en.wikipedia.org/wiki/CAC_40", ["ticker"], ["company"])
    return _standardize(t, ["ticker"], ["company"], ["sector", "gics"], clean_dot=False)


@st.cache_data(ttl=86400, show_spinner=False)
def load_dax40() -> pd.DataFrame:
    t = _best_wiki_table("https://en.wikipedia.org/wiki/DAX", ["ticker symbol", "ticker"], ["company"])
    return _standardize(t, ["ticker symbol", "ticker"], ["company"], ["sector", "industry"], clean_dot=False)


@st.cache_data(ttl=86400, show_spinner=False)
def load_ftse100() -> pd.DataFrame:
    t = _best_wiki_table("https://en.wikipedia.org/wiki/FTSE_100_Index", ["ticker"], ["company"])
    return _standardize(t, ["ticker"], ["company"], ["sector", "industry", "ftse industry"], clean_dot=False)


# ---- Listes curées (explicitement non-officielles, usage illustratif) -----

def load_tech_trending() -> pd.DataFrame:
    data = [
        ("AAPL", "Apple", "Big Tech"), ("MSFT", "Microsoft", "Big Tech"),
        ("GOOGL", "Alphabet", "Big Tech"), ("AMZN", "Amazon", "Big Tech"),
        ("NVDA", "Nvidia", "Semi-conducteurs / IA"), ("META", "Meta Platforms", "Big Tech"),
        ("TSLA", "Tesla", "Tech / Auto"), ("AMD", "AMD", "Semi-conducteurs / IA"),
        ("AVGO", "Broadcom", "Semi-conducteurs / IA"), ("ORCL", "Oracle", "Cloud / IA"),
        ("CRM", "Salesforce", "Cloud / SaaS"), ("ADBE", "Adobe", "Cloud / SaaS"),
        ("NFLX", "Netflix", "Streaming"), ("PLTR", "Palantir", "IA / Data"),
        ("SMCI", "Super Micro Computer", "Infra IA"), ("MU", "Micron", "Semi-conducteurs / IA"),
        ("QCOM", "Qualcomm", "Semi-conducteurs / IA"), ("INTC", "Intel", "Semi-conducteurs / IA"),
        ("IBM", "IBM", "Cloud / IA"), ("NOW", "ServiceNow", "Cloud / SaaS"),
    ]
    return pd.DataFrame(data, columns=["Symbol", "Nom", "Groupe"])


def load_crypto_top() -> pd.DataFrame:
    data = [
        ("BTC-USD", "Bitcoin", "Réserve de valeur"), ("ETH-USD", "Ethereum", "Layer 1"),
        ("BNB-USD", "BNB", "Layer 1 / Exchange"), ("SOL-USD", "Solana", "Layer 1"),
        ("XRP-USD", "XRP", "Paiements"), ("ADA-USD", "Cardano", "Layer 1"),
        ("DOGE-USD", "Dogecoin", "Meme"), ("AVAX-USD", "Avalanche", "Layer 1"),
        ("DOT-USD", "Polkadot", "Interopérabilité"), ("LINK-USD", "Chainlink", "Oracle / Infra"),
        ("LTC-USD", "Litecoin", "Paiements"), ("BCH-USD", "Bitcoin Cash", "Paiements"),
        ("ATOM-USD", "Cosmos", "Interopérabilité"), ("XLM-USD", "Stellar", "Paiements"),
        ("ETC-USD", "Ethereum Classic", "Layer 1"), ("ALGO-USD", "Algorand", "Layer 1"),
        ("VET-USD", "VeChain", "Supply chain"), ("FIL-USD", "Filecoin", "Stockage / Infra"),
        ("ICP-USD", "Internet Computer", "Infra"), ("UNI-USD", "Uniswap", "DeFi"),
        ("AAVE-USD", "Aave", "DeFi"), ("HBAR-USD", "Hedera", "Layer 1"),
        ("NEAR-USD", "NEAR Protocol", "Layer 1"), ("APT-USD", "Aptos", "Layer 1"),
        ("ARB-USD", "Arbitrum", "Layer 2"), ("OP-USD", "Optimism", "Layer 2"),
    ]
    return pd.DataFrame(data, columns=["Symbol", "Nom", "Groupe"])


MARKETS: dict[str, MarketConfig] = {
    "sp500": MarketConfig("sp500", "🇺🇸 S&P 500", load_sp500, "", "$", "Secteur GICS"),
    "nasdaq100": MarketConfig("nasdaq100", "🇺🇸 Nasdaq 100", load_nasdaq100, "", "$", "Secteur GICS"),
    "dow30": MarketConfig("dow30", "🇺🇸 Dow Jones 30", load_dow30, "", "$", "Industrie"),
    "cac40": MarketConfig("cac40", "🇫🇷 CAC 40", load_cac40, ".PA", "€", "Secteur"),
    "dax40": MarketConfig("dax40", "🇩🇪 DAX 40", load_dax40, ".DE", "€", "Secteur"),
    "ftse100": MarketConfig("ftse100", "🇬🇧 FTSE 100", load_ftse100, ".L", "£", "Secteur"),
    "tech_ai": MarketConfig(
        "tech_ai", "🚀 Tech & IA en vogue", load_tech_trending, "", "$", "Thématique", is_curated=True,
        note="Sélection maison de grandes valeurs tech/IA — pas un indice officiel.",
    ),
    "crypto": MarketConfig(
        "crypto", "₿ Cryptomonnaies (Top 26)", load_crypto_top, "", "$", "Catégorie", is_curated=True,
        note="Sélection maison des cryptos majeures — vérifiez que chaque ticker est bien coté sur Yahoo Finance.",
    ),
    "custom": MarketConfig(
        "custom", "🔧 Marché personnalisé", None, "", "$", "Groupe", is_curated=True,
        note="Saisissez vos propres tickers, au format reconnu par Yahoo Finance.",
    ),
}

# ══════════════════════════════════════════════════════════════════════════
# 3. INDICATEURS TECHNIQUES
# ══════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> dict:
    close, volume = df["Close"], df["Volume"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    boll_up, boll_low = sma20 + 2 * std20, sma20 - 2 * std20

    vol_sma20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_sma20.replace(0, np.nan)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    return dict(rsi=rsi, sma20=sma20, boll_up=boll_up, boll_low=boll_low,
                vol_ratio=vol_ratio, macd=macd, macd_signal=macd_signal, macd_hist=macd_hist)


def compute_score(rsi, price, boll_low, vol_ratio, macd_hist_prev, macd_hist_last, pct_from_low) -> int:
    """Score d'opportunité 0-100 combinant survente (RSI), extension des
    Bandes de Bollinger, anomalie de volume, retournement MACD naissant et
    proximité du plus bas sur la période observée."""
    score = 0
    if pd.notna(rsi):
        if rsi < 30:
            score += 30
        elif rsi < 40:
            score += 15
    if pd.notna(boll_low) and price <= boll_low:
        score += 20
    if pd.notna(vol_ratio):
        if vol_ratio > 1.5:
            score += 20
        elif vol_ratio > 1.2:
            score += 10
    if pd.notna(macd_hist_prev) and pd.notna(macd_hist_last) and macd_hist_prev <= 0 and macd_hist_last > 0:
        score += 15  # croisement haussier naissant du MACD
    if pd.notna(pct_from_low):
        if pct_from_low <= 10:
            score += 15
        elif pct_from_low <= 20:
            score += 8
    return int(min(score, 100))


# ══════════════════════════════════════════════════════════════════════════
# 4. PIPELINE DE RÉCUPÉRATION & D'ANALYSE
# ══════════════════════════════════════════════════════════════════════════

def _extract_frame(data: pd.DataFrame, symbol: str, n_tickers: int) -> pd.DataFrame:
    """yfinance ne renvoie pas toujours un MultiIndex quand un seul ticker
    est demandé — on gère les deux cas."""
    if n_tickers > 1 and isinstance(data.columns, pd.MultiIndex):
        return data[symbol].dropna()
    return data.dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_analyze(market_key: str, symbols: list[str], names_map: dict, groups_map: dict):
    """Télécharge l'historique (1 an) de chaque titre et calcule les
    indicateurs + le score d'opportunité. Retourne (DataFrame, erreur|None)."""
    n = len(symbols)
    try:
        data = yf.download(symbols, period="1y", interval="1d", group_by="ticker",
                            auto_adjust=True, threads=True, progress=False)
    except Exception as e:  # réseau, rate-limit, etc.
        return pd.DataFrame(), f"{e}"

    if data is None or data.empty:
        return pd.DataFrame(), "Aucune donnée reçue (marché fermé, tickers invalides ou accès réseau limité)."

    rows = []
    for symbol in symbols:
        try:
            df_s = _extract_frame(data, symbol, n)
            if len(df_s) < 30 or "Close" not in df_s.columns:
                continue

            close = df_s["Close"]
            last_p, prev_p = float(close.iloc[-1]), float(close.iloc[-2])
            var_day = (last_p - prev_p) / prev_p * 100
            var_5d = (last_p - float(close.iloc[-5])) / float(close.iloc[-5]) * 100 if len(close) >= 5 else np.nan

            ind = compute_indicators(df_s)
            rsi = ind["rsi"].iloc[-1]
            boll_low = ind["boll_low"].iloc[-1]
            vol_ratio = ind["vol_ratio"].iloc[-1]
            macd_hist = ind["macd_hist"]
            macd_prev = macd_hist.iloc[-2] if len(macd_hist) >= 2 else np.nan
            macd_last = macd_hist.iloc[-1]

            period_low = float(close.min())
            pct_from_low = (last_p - period_low) / period_low * 100 if period_low else np.nan

            score = compute_score(rsi, last_p, boll_low, vol_ratio, macd_prev, macd_last, pct_from_low)

            rows.append({
                "Ticker": symbol,
                "Nom": names_map.get(symbol, symbol),
                "Groupe": groups_map.get(symbol, "N/A"),
                "Prix": round(last_p, 2),
                "Var. 1J (%)": round(var_day, 2),
                "Var. 5J (%)": round(var_5d, 2) if pd.notna(var_5d) else None,
                "RSI (14)": round(float(rsi), 1) if pd.notna(rsi) else None,
                "Sous Bollinger": bool(pd.notna(boll_low) and last_p <= boll_low),
                "Ratio Vol.": round(float(vol_ratio), 2) if pd.notna(vol_ratio) else None,
                "% vs Bas (période)": round(pct_from_low, 1) if pd.notna(pct_from_low) else None,
                "MACD haussier": bool(pd.notna(macd_prev) and pd.notna(macd_last) and macd_prev <= 0 and macd_last > 0),
                "Score Opp.": score,
                "_history": df_s,
            })
        except Exception:
            continue

    return pd.DataFrame(rows), None


# ══════════════════════════════════════════════════════════════════════════
# 5. SIDEBAR — SÉLECTION DU MARCHÉ & FILTRES
# ══════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🧭 Marché & univers")
market_key = st.sidebar.selectbox(
    "Marché à analyser", list(MARKETS.keys()), format_func=lambda k: MARKETS[k].label,
)
market = MARKETS[market_key]
if market.note:
    st.sidebar.caption(f"ℹ️ {market.note}")

if market.key == "custom":
    raw_input = st.sidebar.text_area(
        "Tickers (séparés par une virgule ou un retour à la ligne)",
        placeholder="AAPL, MSFT, MC.PA, SAP.DE, BTC-USD ...", height=100,
    )
    tickers_raw = sorted({
        t.strip().upper() for chunk in raw_input.replace("\n", ",").split(",") for t in [chunk] if t.strip()
    })
    universe_df = pd.DataFrame({"Symbol": tickers_raw, "Nom": tickers_raw, "Groupe": "Personnalisé"})
else:
    try:
        with st.spinner(f"Chargement de la composition — {market.label}..."):
            universe_df = market.loader()
    except Exception as e:
        st.sidebar.error(f"Impossible de charger la composition du marché : {e}")
        universe_df = pd.DataFrame(columns=["Symbol", "Nom", "Groupe"])

st.sidebar.caption(f"{len(universe_df)} titres dans l'univers sélectionné")
st.sidebar.markdown("---")

st.sidebar.markdown("## 🎯 Profil & filtres")

PRESETS = {
    "🛡️ Conservateur": {"score": 60, "rsi": 30},
    "⚖️ Modéré": {"score": 40, "rsi": 35},
    "🔥 Agressif": {"score": 25, "rsi": 45},
}


def _apply_preset():
    p = PRESETS.get(st.session_state.get("preset_choice"))
    if p:
        st.session_state["min_score"] = p["score"]
        st.session_state["rsi_max"] = p["rsi"]


st.session_state.setdefault("min_score", 40)
st.session_state.setdefault("rsi_max", 35)
st.sidebar.radio(
    "Profil de risque", list(PRESETS.keys()) + ["🎛️ Personnalisé"],
    key="preset_choice", on_change=_apply_preset, index=1,
)

with st.sidebar.expander("Critères de sélection", expanded=True):
    group_options = ["Tous"] + sorted(universe_df["Groupe"].dropna().unique().tolist()) if len(universe_df) else ["Tous"]
    selected_group = st.selectbox(market.group_label, group_options)
    min_score = st.slider("Score Opportunité Min.", 0, 100, step=5, key="min_score")
    rsi_max = st.slider("RSI Max (zone de survente)", 10, 50, key="rsi_max")
    volume_filter = st.checkbox("Volume ≥ moyenne 20 jours (≥ 1.0x)", value=True)
    search_query = st.text_input("🔎 Rechercher un ticker / nom", "")

with st.sidebar.expander("Affichage"):
    max_rows = st.select_slider("Résultats affichés (max.)", options=[10, 20, 50, 100, "Tous"], value=50)
    sort_col = st.selectbox("Trier par", ["Score Opp.", "RSI (14)", "Var. 1J (%)", "Ratio Vol."], index=0)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Rafraîchir les données", width="stretch"):
    fetch_and_analyze.clear()
    st.rerun()
st.sidebar.caption(f"Dernière analyse : {dt.datetime.now().strftime('%d/%m/%Y %H:%M')}")

# ══════════════════════════════════════════════════════════════════════════
# 6. RÉCUPÉRATION DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════
if len(universe_df) == 0:
    st.markdown(
        '<div class="hero"><div class="hero-badge">EN ATTENTE</div>'
        "<h1>Screener d'Opportunités — Court Terme & Rebond</h1>"
        '<p>Sélectionnez un marché ou saisissez des tickers personnalisés dans la barre latérale pour démarrer l\'analyse.</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()

symbols = (universe_df["Symbol"].astype(str) + market.suffix).tolist()
names_map = dict(zip(symbols, universe_df["Nom"]))
groups_map = dict(zip(symbols, universe_df["Groupe"]))

with st.spinner(f"Analyse de {len(symbols)} titres — {market.label}..."):
    results_df, fetch_error = fetch_and_analyze(market.key, symbols, names_map, groups_map)

if fetch_error:
    st.error(f"Erreur lors de la récupération des données : {fetch_error}")
    st.stop()
if results_df.empty:
    st.warning("Aucune donnée exploitable n'a pu être calculée pour cet univers. Réessayez ou changez de marché.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# 7. FILTRAGE
# ══════════════════════════════════════════════════════════════════════════
filtered = results_df.copy()
if selected_group != "Tous":
    filtered = filtered[filtered["Groupe"] == selected_group]
filtered = filtered[
    (filtered["Score Opp."] >= min_score) & (filtered["RSI (14)"].fillna(100) <= rsi_max)
]
if volume_filter:
    filtered = filtered[filtered["Ratio Vol."].fillna(0) >= 1.0]
if search_query:
    q = search_query.lower()
    filtered = filtered[
        filtered["Ticker"].str.lower().str.contains(q) | filtered["Nom"].str.lower().str.contains(q)
    ]
filtered = filtered.sort_values(by=sort_col, ascending=(sort_col == "RSI (14)"))
if max_rows != "Tous":
    filtered = filtered.head(int(max_rows))

# ══════════════════════════════════════════════════════════════════════════
# 8. BANDEAU TICKER TAPE (mouvements du jour, univers complet)
# ══════════════════════════════════════════════════════════════════════════
tape_source = results_df.sort_values("Var. 1J (%)", ascending=False)
tape_items = ""
for _, r in pd.concat([tape_source.head(15), tape_source.tail(10)]).iterrows():
    cls = "tape-up" if r["Var. 1J (%)"] >= 0 else "tape-down"
    arrow = "▲" if r["Var. 1J (%)"] >= 0 else "▼"
    tape_items += f'<span class="tape-item">{r["Ticker"]} <span class="{cls}">{arrow} {r["Var. 1J (%)"]:+.2f}%</span></span>'

st.markdown(
    f'<div class="tape-wrap"><div class="tape-track">{tape_items}{tape_items}</div></div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════
# 9. EN-TÊTE
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-badge">MARCHÉ ACTIF · {market.label}</div>
      <h1>Screener d'Opportunités — Court Terme & Rebond</h1>
      <p>Détection automatisée des titres en survente, avec volume anormal et signaux techniques
      de retournement (RSI, Bandes de Bollinger, MACD) — adaptable à tous les marchés.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════
# 10. KPIs
# ══════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Univers analysé", len(results_df))
k2.metric("Opportunités détectées", len(filtered))
k3.metric("RSI moyen (sélection)", f"{filtered['RSI (14)'].mean():.1f}" if len(filtered) else "—")
k4.metric("Variation moy. 1J", f"{filtered['Var. 1J (%)'].mean():+.2f}%" if len(filtered) else "—")
k5.metric("Score moyen", f"{filtered['Score Opp.'].mean():.0f}/100" if len(filtered) else "—")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# 11. SPOTLIGHT — TOP 3 OPPORTUNITÉS
# ══════════════════════════════════════════════════════════════════════════
if len(filtered) > 0:
    st.markdown("#### 🏆 Meilleures opportunités")
    top3 = filtered.head(3)
    cols = st.columns(len(top3))
    for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
        with col:
            var_cls = "badge-buy" if row["Var. 1J (%)"] >= 0 else "badge-down"
            badges = f'<span class="badge {var_cls}">Var 1J {row["Var. 1J (%)"]:+.2f}%</span>'
            if row["Sous Bollinger"]:
                badges += '<span class="badge badge-warn">Sous Bollinger</span>'
            if row["Ratio Vol."] and row["Ratio Vol."] > 1.2:
                badges += f'<span class="badge badge-neutral">Vol. {row["Ratio Vol."]:.1f}x</span>'
            if row["MACD haussier"]:
                badges += '<span class="badge badge-buy">MACD ↑</span>'

            st.markdown(
                f"""
                <div class="opp-card">
                  <div class="opp-rank">#{i+1} OPPORTUNITÉ</div>
                  <div class="opp-ticker">{row['Ticker']}</div>
                  <div class="opp-name">{row['Nom']}</div>
                  <div class="opp-price">{market.currency}{row['Prix']:.2f}</div>
                  <div>{badges}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=row["Score Opp."],
                number={"suffix": "/100", "font": {"size": 20, "color": "#E7ECF5"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)"},
                    "bar": {"color": "#22C55E" if row["Score Opp."] >= 60 else "#F5A623"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(132,146,172,.15)"},
                        {"range": [40, 70], "color": "rgba(245,166,35,.15)"},
                        {"range": [70, 100], "color": "rgba(34,197,94,.15)"},
                    ],
                },
            ))
            gauge.update_layout(height=110, margin=dict(l=14, r=14, t=6, b=6),
                                 paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E7ECF5"})
            st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False},
                             key=f"gauge_{row['Ticker']}")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# 12. ONGLETS PRINCIPAUX
# ══════════════════════════════════════════════════════════════════════════
tab_table, tab_heat, tab_chart, tab_about = st.tabs(
    ["📋 Sélection", "🗺️ Cartographie sectorielle", "📈 Analyse graphique", "ℹ️ Méthodologie"]
)

# ---- Onglet Sélection -------------------------------------------------
with tab_table:
    if len(filtered) == 0:
        st.warning("Aucune action ne correspond aux critères actuels. Assouplissez les filtres dans la barre latérale.")
    else:
        display_df = filtered.drop(columns=["_history"]).copy()
        display_df["Lien"] = "https://finance.yahoo.com/quote/" + display_df["Ticker"]
        display_df["Sous Bollinger"] = display_df["Sous Bollinger"].map({True: "Oui 🟢", False: "Non"})
        display_df["MACD haussier"] = display_df["MACD haussier"].map({True: "Oui 🟢", False: "—"})

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Score Opp.": st.column_config.ProgressColumn(format="%d/100", min_value=0, max_value=100),
                "Prix": st.column_config.NumberColumn(format=f"{market.currency} %.2f"),
                "Var. 1J (%)": st.column_config.NumberColumn(format="%.2f %%"),
                "Var. 5J (%)": st.column_config.NumberColumn(format="%.2f %%"),
                "Ratio Vol.": st.column_config.NumberColumn(format="%.2f x"),
                "% vs Bas (période)": st.column_config.NumberColumn(format="%.1f %%"),
                "Lien": st.column_config.LinkColumn("Fiche", display_text="Voir ↗"),
            },
        )

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Exporter en CSV",
                data=display_df.drop(columns=["Lien"]).to_csv(index=False).encode("utf-8"),
                file_name=f"opportunites_{market.key}_{dt.date.today()}.csv",
                mime="text/csv",
                width="stretch",
            )
        with c2:
            try:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    display_df.drop(columns=["Lien"]).to_excel(writer, index=False, sheet_name="Opportunités")
                st.download_button(
                    "⬇️ Exporter en Excel",
                    data=buf.getvalue(),
                    file_name=f"opportunites_{market.key}_{dt.date.today()}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )
            except ImportError:
                st.caption("Export Excel indisponible (module openpyxl manquant).")

# ---- Onglet Cartographie sectorielle -----------------------------------
with tab_heat:
    if len(results_df) == 0:
        st.info("Pas de données à cartographier.")
    else:
        heat_df = results_df.copy()
        heat_df["Groupe"] = heat_df["Groupe"].fillna("N/A")
        heat_df["_weight"] = 1  # poids égal par titre (la couleur porte l'information, pas la taille)
        fig = px.treemap(
            heat_df, path=[px.Constant(market.label), "Groupe", "Ticker"], values="_weight",
            color="Var. 1J (%)", color_continuous_scale=["#F0466E", "#232C42", "#22C55E"],
            color_continuous_midpoint=0, hover_data={"Score Opp.": True, "RSI (14)": True, "_weight": False},
        )
        fig.update_traces(texttemplate="<b>%{label}</b><br>%{color:+.2f}%", textposition="middle center")
        fig.update_layout(
            height=560, margin=dict(l=4, r=4, t=30, b=4),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E7ECF5"},
        )
        st.plotly_chart(fig, width="stretch", key="treemap")
        st.caption("Taille = poids égal par titre · Couleur = variation du jour (rouge = baisse, vert = hausse).")

# ---- Onglet Analyse graphique ------------------------------------------
with tab_chart:
    if len(results_df) == 0:
        st.info("Pas de données à afficher.")
    else:
        pool = filtered if len(filtered) > 0 else results_df
        selected_ticker = st.selectbox("Sélectionner un titre :", pool["Ticker"].tolist())
        stock_row = results_df[results_df["Ticker"] == selected_ticker].iloc[0]
        stock_data = stock_row["_history"]
        ind = compute_indicators(stock_data)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            row_heights=[0.55, 0.2, 0.25],
            subplot_titles=(f"{selected_ticker} — Prix & Bandes de Bollinger", "Volume", "RSI (14)"),
        )
        fig.add_trace(go.Candlestick(
            x=stock_data.index, open=stock_data["Open"], high=stock_data["High"],
            low=stock_data["Low"], close=stock_data["Close"], name="Prix",
            increasing_line_color="#22C55E", decreasing_line_color="#F0466E",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=ind["boll_up"], name="Bollinger haut",
                                  line=dict(color="rgba(108,142,255,.5)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=ind["boll_low"], name="Bollinger bas",
                                  line=dict(color="rgba(108,142,255,.5)", width=1),
                                  fill="tonexty", fillcolor="rgba(108,142,255,.06)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=ind["sma20"], name="SMA 20",
                                  line=dict(color="#F5A623", width=1.3)), row=1, col=1)

        vol_colors = np.where(stock_data["Close"] >= stock_data["Open"], "#22C55E", "#F0466E")
        fig.add_trace(go.Bar(x=stock_data.index, y=stock_data["Volume"], name="Volume",
                              marker_color=vol_colors), row=2, col=1)

        fig.add_trace(go.Scatter(x=stock_data.index, y=ind["rsi"], name="RSI (14)",
                                  line=dict(color="#6C8EFF", width=1.5)), row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#22C55E", row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#F0466E", row=3, col=1)

        fig.update_layout(
            height=650, xaxis_rangeslider_visible=False, showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E7ECF5"}, margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        fig.update_xaxes(gridcolor="#232C42")
        fig.update_yaxes(gridcolor="#232C42")
        st.plotly_chart(fig, width="stretch", key="main_chart")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prix", f"{market.currency}{stock_row['Prix']:.2f}", f"{stock_row['Var. 1J (%)']:+.2f}%")
        m2.metric("RSI (14)", stock_row["RSI (14)"])
        m3.metric("Ratio Volume", f"{stock_row['Ratio Vol.']}x" if stock_row["Ratio Vol."] else "—")
        m4.metric("Score Opportunité", f"{stock_row['Score Opp.']}/100")

# ---- Onglet Méthodologie -------------------------------------------------
with tab_about:
    st.markdown("#### Comment le score d'opportunité (0-100) est calculé")
    st.markdown(
        """
        | Signal | Condition | Points |
        |---|---|---|
        | RSI (14) | < 30 (survente forte) | +30 |
        | RSI (14) | entre 30 et 40 (survente modérée) | +15 |
        | Bandes de Bollinger | prix ≤ bande basse (20j, 2σ) | +20 |
        | Volume | ratio vs moyenne 20j > 1.5x | +20 |
        | Volume | ratio vs moyenne 20j entre 1.2x et 1.5x | +10 |
        | MACD | croisement haussier naissant de l'histogramme | +15 |
        | Range période | prix à ≤ 10% du plus bas sur 1 an | +15 |
        | Range période | prix à ≤ 20% du plus bas sur 1 an | +8 |

        Le score est plafonné à 100. Il combine des signaux de **survente**, de **retournement**
        et d'**anomalie de volume** — il ne constitue en aucun cas une garantie de rebond futur.
        """
    )
    st.markdown("#### Marchés disponibles")
    for m in MARKETS.values():
        tag = " · liste maison, non-officielle" if m.is_curated else " · composition officielle (Wikipedia, temps réel)"
        st.markdown(f"- **{m.label}**{tag}")

    st.markdown(
        """
        <p class="footnote">
        Les données proviennent de Yahoo Finance (yfinance) et Wikipedia. Elles peuvent être retardées,
        incomplètes ou erronées. Cet outil est fourni à titre informatif et pédagogique uniquement —
        il ne constitue ni un conseil en investissement, ni une recommandation d'achat ou de vente.
        Faites toujours vos propres recherches (DYOR) et consultez un professionnel agréé avant toute décision.
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="footnote" style="margin-top:24px;">⚠️ Outil pédagogique — ne constitue pas un conseil en investissement.</p>',
    unsafe_allow_html=True,
)
