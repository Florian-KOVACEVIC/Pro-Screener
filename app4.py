"""
================================================================================
 PRO SCREENER : Opportunités Court Terme et Rebond (Multi-Marchés)
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
import math
import re
import unicodedata
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
from PIL import Image, ImageDraw
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION DE PAGE & STYLE
# ══════════════════════════════════════════════════════════════════════════

def _build_favicon() -> Image.Image:
    """Construit l'icône d'onglet du navigateur sous forme d'image bitmap,
    avec le même dégradé rouge/orangé que la police des titres.

    Un simple caractère Unicode ('▣') passé à page_icon dépend du support
    emoji/police du navigateur et ne s'affiche pas de façon fiable partout :
    on dessine donc l'icône nous-mêmes, ce qui garantit un rendu identique
    dans tous les navigateurs. Cela n'a aucun lien avec le bandeau des
    mouvements du marché affiché dans la page (le "ticker tape").
    """
    size = 64
    stops = [(240, 70, 110), (255, 106, 69), (245, 166, 35)]  # rose -> orange -> ambre
    grad_row = Image.new("RGB", (size, 1))
    for x in range(size):
        t = x / (size - 1)
        if t <= 0.45:
            lt = t / 0.45
            c = tuple(int(stops[0][i] + (stops[1][i] - stops[0][i]) * lt) for i in range(3))
        else:
            lt = (t - 0.45) / 0.55
            c = tuple(int(stops[1][i] + (stops[2][i] - stops[1][i]) * lt) for i in range(3))
        grad_row.putpixel((x, 0), c)
    gradient = grad_row.resize((size, size))

    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)
    margin = 4
    mdraw.rounded_rectangle([margin, margin, size - margin, size - margin], radius=12, outline=255, width=7)
    inner = size * 0.30
    cx = cy = size / 2
    mdraw.rectangle([cx - inner / 2, cy - inner / 2, cx + inner / 2, cy + inner / 2], fill=255)

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    img.paste(gradient, (0, 0), mask)
    return img

def render_hero_icon_svg(size: int = 40) -> str:
    """Icône affichée à côté du titre principal, avec le même dégradé que
    le favicon et que le texte en dégradé (.gradient-text).

    Rendue sur une seule ligne :
    lorsque cette chaîne est insérée au milieu d'un autre bloc HTML passé à
    st.markdown, une ligne vide ferait croire au moteur Markdown que le bloc
    HTML brut est terminé, et le reste (le titre) serait alors affiché comme
    du texte échappé au lieu d'être interprété comme du HTML.
    """
    return (
        f'<svg class="hero-icon" width="{size}" height="{size}" viewBox="0 0 64 64" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<defs><linearGradient id="heroIconGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
        '<stop offset="0%" stop-color="#F0466E"/><stop offset="45%" stop-color="#FF6A45"/>'
        '<stop offset="100%" stop-color="#F5A623"/></linearGradient></defs>'
        '<rect x="5" y="5" width="54" height="54" rx="12" fill="none" stroke="url(#heroIconGrad)" stroke-width="7"/>'
        '<rect x="24" y="24" width="16" height="16" fill="url(#heroIconGrad)"/></svg>'
    )

st.set_page_config(
    page_title="Pro Screener, Multi-Marchés",
    page_icon=_build_favicon(),
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

        html, body, [class*="css"]{ font-family:'Inter',sans-serif; color-scheme:dark; }
        html, body,
        [data-testid="stAppViewContainer"], [data-testid="stApp"], [data-testid="stMain"],
        [data-testid="stHeader"], [data-testid="stBottomBlockContainer"], .main{
            background-color:var(--bg-void) !important; color:var(--text-hi) !important;
        }
        .stApp{ background:radial-gradient(ellipse at top, #0D1220 0%, var(--bg-void) 55%) !important; }
        /* Menus/popovers BaseWeb (selectbox, select multiple) sont montés hors de l'arbre DOM normal :
           on les cible explicitement pour éviter tout résidu de thème clair. */
        [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], ul[data-testid="stSelectboxVirtualDropdown"]{
            background-color:var(--bg-card) !important; border-color:var(--border) !important;
        }
        [data-baseweb="menu"] li, [role="option"]{ color:var(--text-hi) !important; }

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
        .hero-title-row{ display:flex; align-items:center; gap:14px; }
        .hero-icon{ flex-shrink:0; }
        .hero h1{ font-size:2.1rem; margin:0; }
        .hero p{ color:var(--text-lo); font-size:.98rem; max-width:760px; margin:10px 0 0 0; }

        /* ---------- Titres en dégradé ---------- */
        .gradient-text{
            background:linear-gradient(90deg,#F0466E 0%,#FF6A45 45%,#F5A623 100%);
            -webkit-background-clip:text; background-clip:text; color:transparent !important;
            display:inline-block;
        }

        /* ---------- Titres de la barre latérale ---------- */
        .sidebar-title{ font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:700;
                          margin:4px 0 10px 0;
                          background:linear-gradient(90deg,#F0466E 0%,#FF6A45 45%,#F5A623 100%);
                          -webkit-background-clip:text; background-clip:text; color:transparent !important;
                          display:inline-block; }

        /* ---------- Disclaimer centré et discret ---------- */
        .disclaimer{ max-width:680px; margin:40px auto 12px auto; text-align:center;
                      color:var(--text-lo); font-size:.74rem; line-height:1.6; opacity:.75; }

        /* ---------- Grille de KPI (responsive : passe à la ligne si l'écran est étroit) ---------- */
        .kpi-grid{ display:flex; flex-wrap:wrap; gap:12px; margin:4px 0 8px 0; }
        .kpi-card{ flex:1 1 170px; min-width:150px; background:var(--bg-card); border:1px solid var(--border);
                    border-radius:14px; padding:14px 18px; transition:box-shadow .18s ease, border-color .18s ease,
                    transform .18s ease; cursor:default; }
        .kpi-card:hover{ border-color:transparent; transform:translateY(-1px);
                    box-shadow:0 0 0 1.5px #F0466E, 0 0 0 1.5px #F5A623 inset, 0 10px 26px -10px rgba(240,70,110,.5); }
        .kpi-label{ color:var(--text-lo); font-size:.8rem; margin-bottom:6px; white-space:normal; }
        .kpi-value{ font-family:'IBM Plex Mono',monospace; color:var(--text-hi); font-size:1.35rem; font-weight:700;
                     white-space:normal; word-break:break-word; line-height:1.2; }

        /* ---------- Survol dégradé rouge -> orange, réutilisé sur les cases importantes ---------- */
        div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button,
        div[data-testid="stFormSubmitButton"] > button{
            transition:box-shadow .18s ease, border-color .18s ease, color .18s ease;
        }
        div[data-testid="stButton"] > button:hover, div[data-testid="stDownloadButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover{
            border-color:transparent !important; color:var(--text-hi) !important;
            box-shadow:0 0 0 1.5px #F0466E, 0 0 0 1.5px #F5A623 inset, 0 8px 20px -10px rgba(240,70,110,.5) !important;
        }

        /* ---------- Spotlight cards : bloc HTML auto-porté (bordure/fond/hover intégrés),
           plus fiable qu'un ciblage CSS du conteneur Streamlit englobant. ---------- */
        .opp-card-shell{ background:var(--bg-card); border:1px solid var(--border); border-radius:16px;
                    padding:16px 18px 18px 18px; transition:box-shadow .18s ease, border-color .18s ease,
                    transform .18s ease; cursor:default; }
        .opp-card-shell:hover{ border-color:transparent; transform:translateY(-2px);
                    box-shadow:0 0 0 1.5px #F0466E, 0 0 0 1.5px #F5A623 inset, 0 14px 32px -12px rgba(240,70,110,.55); }
        .opp-rank{ font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--text-lo); text-transform:uppercase; letter-spacing:.08em; }
        .opp-name{ font-family:'Space Grotesk',sans-serif; font-size:1.2rem; font-weight:700; color:var(--text-hi); margin:2px 0 0 0; line-height:1.25; }
        .opp-ticker{ font-family:'IBM Plex Mono',monospace; font-size:.8rem; color:var(--text-lo); letter-spacing:.03em; margin-bottom:10px; }
        .opp-price{ font-family:'IBM Plex Mono',monospace; font-size:1.55rem; font-weight:700; color:var(--text-hi); margin-bottom:10px; }

        /* ---------- Panneaux (graphiques / tableaux) ---------- */
        .panel-title{ font-family:'Space Grotesk',sans-serif; font-size:1.05rem; font-weight:700;
                        margin:0 0 3px 0; padding-left:11px; border-left:3px solid var(--emerald);
                        background:linear-gradient(90deg,#F0466E 0%,#FF6A45 45%,#F5A623 100%);
                        -webkit-background-clip:text; background-clip:text; color:transparent !important;
                        display:inline-block; }
        .panel-sub{ color:var(--text-lo); font-size:.8rem; margin:0 0 16px 14px; }
        [data-testid="stVerticalBlockBorderWrapper"]{ border-radius:16px !important; }

        /* ---------- Jauge de score (SVG, centrage garanti à toute taille) ---------- */
        .gauge-wrap{ width:100%; display:flex; justify-content:center; align-items:center; margin-top:8px; }
        .gauge-svg{ width:100%; max-width:200px; height:auto; display:block; }
        .gauge-value{ font-family:'IBM Plex Mono',monospace; font-size:36px; font-weight:700; fill:var(--text-hi); }
        .gauge-suffix{ font-family:'IBM Plex Mono',monospace; font-size:13px; fill:var(--text-lo); }

        .badge{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:600;
                 font-family:'IBM Plex Mono',monospace; margin-right:6px; margin-top:6px; }
        .badge-buy{ background:var(--emerald-soft); color:var(--emerald); }
        .badge-warn{ background:var(--amber-soft); color:var(--amber); }
        .badge-neutral{ background:rgba(132,146,172,.14); color:var(--text-lo); }
        .badge-down{ background:var(--rose-soft); color:var(--rose); }

        /* ---------- Dataframe / tabs ---------- */
        [data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:12px; overflow:hidden; }
        .stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--border); }
        .stTabs [data-baseweb="tab"]{ font-family:'Space Grotesk',sans-serif; font-weight:600; color:var(--text-lo); }
        .stTabs [aria-selected="true"] p,
        .stTabs [aria-selected="true"] [data-testid="stMarkdownContainer"]{
            background:linear-gradient(90deg,#F0466E 0%,#FF6A45 45%,#F5A623 100%);
            -webkit-background-clip:text; background-clip:text; color:transparent !important;
            display:inline-block; font-weight:700;
        }

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

def _flatten_cols(df: pd.DataFrame) -> list[str]:
    """Aplati les colonnes (gère le cas où Wikipedia ajoute une ligne d'en-tête
    supplémentaire, ce qui fait lire les colonnes comme un MultiIndex par
    pandas.read_html au lieu de simples chaînes)."""
    flat = []
    for c in df.columns:
        if isinstance(c, tuple):
            flat.append(" ".join(str(p) for p in c if str(p) != "nan"))
        else:
            flat.append(str(c))
    return flat

def _looks_unnamed(cols: list[str]) -> bool:
    unnamed = sum(1 for c in cols if c.lower().startswith("unnamed"))
    return unnamed >= max(1, len(cols) // 2)

def _repair_header(t: pd.DataFrame) -> pd.DataFrame:
    """Si les colonnes ressemblent à des noms auto-générés par pandas
    ('Unnamed: 0', 'Unnamed: 1'...), Wikipedia a probablement un en-tête
    étalé sur plusieurs lignes que pandas n'a pas su fusionner correctement :
    on promeut la première ligne de données en en-tête et on retente."""
    if len(t) > 1 and _looks_unnamed(_flatten_cols(t)):
        repaired = t.iloc[1:].copy()
        repaired.columns = [str(c) for c in t.iloc[0]]
        return repaired
    return t

def _best_wiki_table(url: str, ticker_keys: list[str], name_keys: list[str]) -> pd.DataFrame:
    """Récupère la page Wikipedia et retourne la table de composants la plus
    probable : parmi toutes les tables contenant à la fois une colonne
    'ticker-like' et une colonne 'nom-like', on garde la plus grande (le
    nombre de lignes), pour éviter qu'une petite table annexe (ex :
    'changements récents') ne soit choisie par erreur. Robuste aussi aux
    en-têtes multi-lignes (MultiIndex, ou en-têtes que pandas ne détecte pas
    du tout et remplace par des noms 'Unnamed: N')."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    try:
        tables = pd.read_html(io.StringIO(resp.text), attrs={"class": "wikitable"})
    except ValueError:
        tables = []
    if not tables:
        tables = pd.read_html(io.StringIO(resp.text))

    candidates = []
    for t in tables:
        for candidate in (t, _repair_header(t)):
            cols = [c.lower() for c in _flatten_cols(candidate)]
            has_ticker = any(any(k in c for k in ticker_keys) for c in cols)
            has_name = any(any(k in c for k in name_keys) for c in cols)
            if has_ticker and has_name and len(candidate) >= 5:
                candidates.append(candidate)
                break
    if not candidates:
        raise ValueError(
            f"table des composants introuvable sur la page Wikipedia ({len(tables)} table(s) "
            "analysée(s), aucune ne correspond aux colonnes attendues)."
        )
    return max(candidates, key=len)

def _col(df: pd.DataFrame, keys: list[str]) -> Optional[str]:
    for c in df.columns:
        if any(k in str(c).lower() for k in keys):
            return c
    return None

def _clean_cell(series: pd.Series) -> pd.Series:
    """Nettoie une colonne extraite de Wikipedia : notes de bas de page
    ('MC[1]', 'MC†'), retours à la ligne parasites, espaces multiples."""
    s = series.astype(str)
    s = s.str.replace(r"\[.*?\]", "", regex=True)
    s = s.str.replace(r"[\n\r\t]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.str.strip()

def _standardize(df: pd.DataFrame, ticker_keys, name_keys, sector_keys, clean_dot=True) -> pd.DataFrame:
    tcol, ncol, scol = _col(df, ticker_keys), _col(df, name_keys), _col(df, sector_keys)
    out = pd.DataFrame()
    out["Symbol"] = _clean_cell(df[tcol])
    if clean_dot:
        out["Symbol"] = out["Symbol"].str.replace(".", "-", regex=False)  # ex: BRK.B -> BRK-B
    out["Nom"] = _clean_cell(df[ncol]) if ncol else out["Symbol"]
    out["Groupe"] = _clean_cell(df[scol]) if scol else "N/A"
    out = out[out["Symbol"].str.len() > 0]
    out = out[~out["Symbol"].str.lower().isin(["nan", "none"])]
    return out.drop_duplicates(subset="Symbol").reset_index(drop=True)

def _ensure_suffix(ticker: str, suffix: str) -> str:
    """Évite un double suffixe (ex: 'MC.PA' + '.PA' -> 'MC.PA.PA') si la
    colonne source contenait déjà le suffixe de la place boursière."""
    if not suffix:
        return ticker
    return ticker if ticker.upper().endswith(suffix.upper()) else ticker + suffix

# ---- Chargeurs d'indices officiels (scraping Wikipedia en direct) ---------
# NB : par prudence, aucune liste de secours codée en dur n'est utilisée pour
# les indices officiels ci-dessous : une composition d'indice inventee ou
# obsolète serait trompeuse. En cas d'échec du scraping, l'app affiche une
# erreur claire plutôt que de substituer des données non fiables.

@st.cache_data(ttl=86400, show_spinner=False)
def load_sp500() -> pd.DataFrame:
    t = _best_wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["symbol"], ["security", "company"])
    return _standardize(t, ["symbol"], ["security", "company"], ["gics sector", "sector"])

@st.cache_data(ttl=86400, show_spinner=False)
def load_nasdaq100() -> pd.DataFrame:
    """Composition codée en dur plutôt que scrapée en direct : la page
    Wikipedia du Nasdaq-100 s'est révélée trop instable à parser de façon
    fiable (en-têtes multi-lignes, structure changeante). Base de confiance
    moyenne à haute sur les valeurs les plus importantes, plus faible sur la
    longue traîne (l'indice compte 100 valeurs avec un renouvellement annuel
    non négligeable) : si un titre est signalé dans le panneau "titres
    ignorés" au chargement, son ticker a probablement changé depuis."""
    data = [
        ("AAPL", "Apple", "Tech"), ("MSFT", "Microsoft", "Tech"), ("GOOGL", "Alphabet (A)", "Tech"),
        ("GOOG", "Alphabet (C)", "Tech"), ("AMZN", "Amazon", "Conso discrétionnaire"),
        ("NVDA", "Nvidia", "Semi-conducteurs"), ("META", "Meta Platforms", "Tech"),
        ("TSLA", "Tesla", "Automobile"), ("AVGO", "Broadcom", "Semi-conducteurs"),
        ("COST", "Costco", "Distribution"), ("PEP", "PepsiCo", "Consommation"),
        ("ADBE", "Adobe", "Logiciel"), ("NFLX", "Netflix", "Media / Streaming"),
        ("AMD", "AMD", "Semi-conducteurs"), ("CSCO", "Cisco", "Réseaux"),
        ("TMUS", "T-Mobile US", "Télécoms"), ("INTC", "Intel", "Semi-conducteurs"),
        ("QCOM", "Qualcomm", "Semi-conducteurs"), ("TXN", "Texas Instruments", "Semi-conducteurs"),
        ("AMGN", "Amgen", "Biotech"), ("HON", "Honeywell", "Industrie"),
        ("INTU", "Intuit", "Logiciel"), ("AMAT", "Applied Materials", "Equipements semi-conducteurs"),
        ("BKNG", "Booking Holdings", "Voyage / Internet"), ("ISRG", "Intuitive Surgical", "Santé / Robotique"),
        ("VRTX", "Vertex Pharmaceuticals", "Biotech"), ("ADP", "ADP", "Services aux entreprises"),
        ("SBUX", "Starbucks", "Restauration"), ("GILD", "Gilead Sciences", "Biotech"),
        ("MU", "Micron", "Semi-conducteurs"), ("LRCX", "Lam Research", "Equipements semi-conducteurs"),
        ("ADI", "Analog Devices", "Semi-conducteurs"), ("PANW", "Palo Alto Networks", "Cybersécurité"),
        ("MDLZ", "Mondelez International", "Consommation"), ("REGN", "Regeneron", "Biotech"),
        ("KLAC", "KLA Corp", "Equipements semi-conducteurs"), ("SNPS", "Synopsys", "Logiciel"),
        ("CDNS", "Cadence Design Systems", "Logiciel"), ("MELI", "Mercado Libre", "E-commerce"),
        ("CSX", "CSX Corp", "Transport ferroviaire"), ("MAR", "Marriott International", "Hôtellerie"),
        ("ORLY", "O'Reilly Automotive", "Distribution"), ("CTAS", "Cintas", "Services aux entreprises"),
        ("ASML", "ASML Holding", "Equipements semi-conducteurs"), ("PYPL", "PayPal", "Paiements"),
        ("NXPI", "NXP Semiconductors", "Semi-conducteurs"), ("ABNB", "Airbnb", "Voyage / Internet"),
        ("WDAY", "Workday", "Logiciel"), ("FTNT", "Fortinet", "Cybersécurité"),
        ("MNST", "Monster Beverage", "Consommation"), ("PCAR", "Paccar", "Industrie / Camions"),
        ("ROP", "Roper Technologies", "Industrie"), ("PAYX", "Paychex", "Services aux entreprises"),
        ("AEP", "American Electric Power", "Energie / Utilities"), ("ODFL", "Old Dominion Freight Line", "Transport"),
        ("KDP", "Keurig Dr Pepper", "Consommation"), ("EXC", "Exelon", "Energie / Utilities"),
        ("CPRT", "Copart", "Services aux entreprises"), ("DXCM", "Dexcom", "Santé / Medtech"),
        ("XEL", "Xcel Energy", "Energie / Utilities"), ("CRWD", "CrowdStrike", "Cybersécurité"),
        ("ROST", "Ross Stores", "Distribution"), ("FAST", "Fastenal", "Distribution industrielle"),
        ("IDXX", "IDEXX Laboratories", "Santé animale"), ("VRSK", "Verisk Analytics", "Data / Analytics"),
        ("BIIB", "Biogen", "Biotech"), ("EA", "Electronic Arts", "Jeu vidéo"),
        ("GEHC", "GE HealthCare", "Santé / Medtech"), ("CTSH", "Cognizant", "Services IT"),
        ("DDOG", "Datadog", "Logiciel / Cloud"), ("TTWO", "Take-Two Interactive", "Jeu vidéo"),
        ("ANSS", "Ansys", "Logiciel"), ("ON", "ON Semiconductor", "Semi-conducteurs"),
        ("GFS", "GlobalFoundries", "Semi-conducteurs"), ("ZS", "Zscaler", "Cybersécurité"),
        ("TEAM", "Atlassian", "Logiciel"), ("ILMN", "Illumina", "Biotech / Séquençage"),
        ("MRVL", "Marvell Technology", "Semi-conducteurs"), ("LULU", "Lululemon", "Habillement"),
        ("SIRI", "Sirius XM", "Media"), ("DASH", "DoorDash", "Livraison / Internet"),
        ("CDW", "CDW Corp", "Distribution IT"), ("FANG", "Diamondback Energy", "Energie"),
        ("MCHP", "Microchip Technology", "Semi-conducteurs"), ("CHTR", "Charter Communications", "Télécoms"),
        ("KHC", "Kraft Heinz", "Consommation"), ("CCEP", "Coca-Cola Europacific Partners", "Consommation"),
        ("WBD", "Warner Bros Discovery", "Media"), ("EBAY", "eBay", "E-commerce"),
        ("APP", "AppLovin", "Logiciel / Adtech"), ("AXON", "Axon Enterprise", "Sécurité / Défense"),
    ]
    return pd.DataFrame(data, columns=["Symbol", "Nom", "Groupe"])

@st.cache_data(ttl=86400, show_spinner=False)
def load_dow30() -> pd.DataFrame:
    """Composition codée en dur (indice très stable, peu de rotations par
    an) plutôt que scrapée en direct, pour la même raison de fiabilité que
    le Nasdaq-100 : la page Wikipedia s'est révélée instable à parser."""
    data = [
        ("AAPL", "Apple", "Tech"), ("MSFT", "Microsoft", "Tech"), ("UNH", "UnitedHealth Group", "Santé"),
        ("GS", "Goldman Sachs", "Finance"), ("HD", "Home Depot", "Distribution"),
        ("CAT", "Caterpillar", "Industrie"), ("CRM", "Salesforce", "Logiciel"),
        ("MCD", "McDonald's", "Restauration"), ("V", "Visa", "Paiements"),
        ("AMGN", "Amgen", "Biotech"), ("TRV", "Travelers Companies", "Assurance"),
        ("AXP", "American Express", "Finance"), ("JPM", "JPMorgan Chase", "Finance"),
        ("IBM", "IBM", "Tech"), ("HON", "Honeywell", "Industrie"),
        ("PG", "Procter & Gamble", "Consommation"), ("CVX", "Chevron", "Energie"),
        ("BA", "Boeing", "Aéronautique"), ("NKE", "Nike", "Habillement"),
        ("JNJ", "Johnson & Johnson", "Santé"), ("MRK", "Merck", "Pharma"),
        ("DIS", "Walt Disney", "Media / Divertissement"), ("KO", "Coca-Cola", "Consommation"),
        ("MMM", "3M", "Industrie"), ("WMT", "Walmart", "Distribution"),
        ("NVDA", "Nvidia", "Semi-conducteurs"), ("SHW", "Sherwin-Williams", "Chimie"),
        ("AMZN", "Amazon", "Conso discrétionnaire"), ("CSCO", "Cisco", "Réseaux"),
        ("VZ", "Verizon", "Télécoms"),
    ]
    return pd.DataFrame(data, columns=["Symbol", "Nom", "Groupe"])

@st.cache_data(ttl=86400, show_spinner=False)
def load_ftse100() -> pd.DataFrame:
    t = _best_wiki_table("https://en.wikipedia.org/wiki/FTSE_100_Index", ["ticker", "epic"], ["company", "name"])
    return _standardize(t, ["ticker", "epic"], ["company", "name"], ["sector", "industry", "ftse industry"], clean_dot=False)

@st.cache_data(ttl=86400, show_spinner=False)
def load_hangseng() -> pd.DataFrame:
    t = _best_wiki_table("https://en.wikipedia.org/wiki/Hang_Seng_Index", ["ticker", "sehk", "code", "symbol"], ["constituent", "company", "name"])
    df = _standardize(t, ["ticker", "sehk", "code", "symbol"], ["constituent", "company", "name"], ["sector", "industry"], clean_dot=False)
    df["Symbol"] = df["Symbol"].str.extract(r"(\d+)")[0].str.zfill(4)
    return df.dropna(subset=["Symbol"]).drop_duplicates(subset="Symbol").reset_index(drop=True)

# ---- Listes curées (explicitement non-officielles, usage illustratif) -----
# CAC 40, DAX 40 et Nikkei 225 sont ici en liste curée plutôt qu'en scraping
# Wikipedia en direct : ces trois pages se sont montrées peu fiables en
# pratique (tableaux absents, colonnes ambiguës, tickers mal formés
# provoquant l'échec du téléchargement des cours). Une liste maison stable
# et honnêtement annoncée comme non exhaustive est préférable à une source
# qui échoue une fois sur deux.

def load_cac40_leaders() -> pd.DataFrame:
    data = [
        ("MC.PA", "LVMH", "Luxe"), ("OR.PA", "L'Oréal", "Consommation"),
        ("TTE", "TotalEnergies", "Énergie"), ("SAN.PA", "Sanofi", "Santé"),
        ("BNP.PA", "BNP Paribas", "Finance"), ("CS.PA", "AXA", "Finance"),
        ("AI.PA", "Air Liquide", "Industrie"), ("AIR.PA", "Airbus", "Aéronautique"),
        ("BN.PA", "Danone", "Consommation"), ("RNO.PA", "Renault", "Automobile"),
        ("CA.PA", "Carrefour", "Distribution"), ("DG.PA", "Vinci", "BTP"),
        ("KER.PA", "Kering", "Luxe"), ("SU.PA", "Schneider Electric", "Industrie"),
        ("SGO.PA", "Saint-Gobain", "Matériaux"), ("EL.PA", "EssilorLuxottica", "Santé"),
        ("RMS.PA", "Hermès", "Luxe"), ("GLE.PA", "Société Générale", "Finance"),
        ("ACA.PA", "Crédit Agricole", "Finance"), ("EN.PA", "Bouygues", "BTP / Télécom"),
        ("ML.PA", "Michelin", "Industrie"), ("HO.PA", "Thales", "Défense"),
        ("SAF.PA", "Safran", "Aéronautique"), ("PUB.PA", "Publicis", "Communication"),
        ("LR.PA", "Legrand", "Industrie"), ("CAP.PA", "Capgemini", "Tech / Conseil"),
        ("DSY.PA", "Dassault Systèmes", "Tech"), ("RI.PA", "Pernod Ricard", "Consommation"),
        ("VIE.PA", "Veolia", "Environnement"), ("ENGI.PA", "Engie", "Énergie"),
        ("ORA.PA", "Orange", "Télécom"), ("STM", "STMicroelectronics", "Semi-conducteurs"),
    ]
    return pd.DataFrame(data, columns=["Symbol", "Nom", "Groupe"])

def load_dax40_leaders() -> pd.DataFrame:
    data = [
        ("SAP", "SAP", "Logiciels"), ("SIE.DE", "Siemens", "Industrie"),
        ("VOW3.DE", "Volkswagen", "Automobile"), ("BMW.DE", "BMW", "Automobile"),
        ("MBG.DE", "Mercedes-Benz Group", "Automobile"), ("ALV.DE", "Allianz", "Finance"),
        ("ADS.DE", "Adidas", "Consommation"), ("DBK.DE", "Deutsche Bank", "Finance"),
        ("BAS.DE", "BASF", "Chimie"), ("BAYN.DE", "Bayer", "Santé / Chimie"),
        ("IFX.DE", "Infineon", "Semi-conducteurs"), ("DTE.DE", "Deutsche Telekom", "Télécom"),
        ("MUV2.DE", "Munich Re", "Assurance"), ("RWE.DE", "RWE", "Énergie"),
        ("EOAN.DE", "E.ON", "Énergie"), ("DHL.DE", "Deutsche Post DHL Group", "Logistique"),
        ("CON.DE", "Continental", "Automobile"), ("HEN3.DE", "Henkel", "Consommation"),
        ("FRE.DE", "Fresenius", "Santé"), ("VNA.DE", "Vonovia", "Immobilier"),
    ]
    return pd.DataFrame(data, columns=["Symbol", "Nom", "Groupe"])

def load_nikkei_leaders() -> pd.DataFrame:
    data = [
        ("7203.T", "Toyota Motor", "Automobile"), ("6758.T", "Sony Group", "Électronique"),
        ("9984.T", "SoftBank Group", "Tech / Investissement"), ("6861.T", "Keyence", "Électronique"),
        ("9983.T", "Fast Retailing (Uniqlo)", "Distribution"), ("8035.T", "Tokyo Electron", "Semi-conducteurs"),
        ("7974.T", "Nintendo", "Jeux vidéo"), ("8306.T", "Mitsubishi UFJ Financial Group", "Finance"),
        ("7267.T", "Honda Motor", "Automobile"), ("6501.T", "Hitachi", "Industrie"),
        ("6752.T", "Panasonic", "Électronique"), ("7751.T", "Canon", "Électronique"),
        ("8058.T", "Mitsubishi Corporation", "Négoce"), ("8031.T", "Mitsui & Co", "Négoce"),
        ("4063.T", "Shin-Etsu Chemical", "Chimie"), ("6367.T", "Daikin Industries", "Industrie"),
        ("9433.T", "KDDI", "Télécom"), ("9432.T", "Nippon Telegraph and Telephone", "Télécom"),
        ("6098.T", "Recruit Holdings", "Services"), ("6981.T", "Murata Manufacturing", "Électronique"),
        ("4502.T", "Takeda Pharmaceutical", "Santé"), ("7201.T", "Nissan Motor", "Automobile"),
    ]
    return pd.DataFrame(data, columns=["Symbol", "Nom", "Groupe"])

def load_sx5e_leaders() -> pd.DataFrame:
    """Grandes valeurs de la zone euro proches de la composition du
    EURO STOXX 50 (SX5E). Sélection maison, non exhaustive."""
    data = [
        ("MC.PA", "LVMH", "Luxe"), ("TTE", "TotalEnergies", "Énergie"),
        ("SAN.PA", "Sanofi", "Santé"), ("SAP", "SAP", "Logiciels"),
        ("SIE.DE", "Siemens", "Industrie"), ("ALV.DE", "Allianz", "Finance"),
        ("ASML", "ASML Holding", "Semi-conducteurs"), ("AIR.PA", "Airbus", "Aéronautique"),
        ("AI.PA", "Air Liquide", "Industrie"), ("SAN.MC", "Banco Santander", "Finance"),
        ("IBE.MC", "Iberdrola", "Énergie"), ("ISP.MI", "Intesa Sanpaolo", "Finance"),
        ("ENEL.MI", "Enel", "Énergie"), ("ENI.MI", "Eni", "Énergie"),
        ("DTE.DE", "Deutsche Telekom", "Télécom"), ("MUV2.DE", "Munich Re", "Assurance"),
        ("VOW3.DE", "Volkswagen", "Automobile"), ("BMW.DE", "BMW", "Automobile"),
        ("DG.PA", "Vinci", "BTP"), ("KER.PA", "Kering", "Luxe"),
        ("BNP.PA", "BNP Paribas", "Finance"), ("CS.PA", "AXA", "Finance"),
        ("SU.PA", "Schneider Electric", "Industrie"), ("ADS.DE", "Adidas", "Consommation"),
        ("BN.PA", "Danone", "Consommation"), ("SAF.PA", "Safran", "Aéronautique"),
    ]
    return pd.DataFrame(data, columns=["Symbol", "Nom", "Groupe"])

def load_kospi_leaders() -> pd.DataFrame:
    """Sélection maison des plus grandes capitalisations cotées au KOSPI.
    Non exhaustive : Wikipedia ne publie pas de tableau complet et fiable des
    ~800 composants du KOSPI, contrairement au Nikkei 225 ou au Hang Seng.
    Confiance moindre que les autres listes maison sur les tickers les plus
    récents (scissions/holdings type SK Square, HD Hyundai) : si l'un d'eux
    est ignoré au chargement (panneau "titres ignorés"), le ticker a
    probablement changé depuis, à vérifier sur Yahoo Finance."""
    data = [
        ("005930.KS", "Samsung Electronics", "Semi-conducteurs / Electronique"),
        ("000660.KS", "SK Hynix", "Semi-conducteurs"),
        ("373220.KS", "LG Energy Solution", "Batteries / Energie"),
        ("207940.KS", "Samsung Biologics", "Santé / Biotech"),
        ("005380.KS", "Hyundai Motor", "Automobile"),
        ("000270.KS", "Kia", "Automobile"),
        ("006400.KS", "Samsung SDI", "Batteries / Energie"),
        ("051910.KS", "LG Chem", "Chimie / Batteries"),
        ("005490.KS", "POSCO Holdings", "Acier / Matériaux"),
        ("035420.KS", "Naver", "Internet"),
        ("035720.KS", "Kakao", "Internet"),
        ("068270.KS", "Celltrion", "Santé / Biotech"),
        ("105560.KS", "KB Financial Group", "Finance"),
        ("055550.KS", "Shinhan Financial Group", "Finance"),
        ("028260.KS", "Samsung C&T", "Conglomérat"),
        ("032830.KS", "Samsung Life Insurance", "Assurance"),
        ("012330.KS", "Hyundai Mobis", "Automobile / Equipementier"),
        ("096770.KS", "SK Innovation", "Energie"),
        ("017670.KS", "SK Telecom", "Télécoms"),
        ("030200.KS", "KT Corp", "Télécoms"),
        ("086790.KS", "Hana Financial Group", "Finance"),
        ("015760.KS", "Korea Electric Power (KEPCO)", "Energie / Utilities"),
        ("066570.KS", "LG Electronics", "Electronique grand public"),
        ("003550.KS", "LG Corp (holding)", "Conglomérat"),
        ("090430.KS", "Amorepacific", "Cosmétiques"),
        ("010130.KS", "Korea Zinc", "Mines / Métaux"),
        ("021240.KS", "Coway", "Biens de consommation"),
        ("034220.KS", "LG Display", "Electronique / Ecrans"),
        ("316140.KS", "Woori Financial Group", "Finance"),
        ("352820.KS", "Hybe", "Divertissement / K-pop"),
    ]
    return pd.DataFrame(data, columns=["Symbol", "Nom", "Groupe"])

def load_asia_tech_leaders() -> pd.DataFrame:
    """Sélection maison de grandes valeurs technologiques asiatiques
    (Japon, Corée, Taïwan, Chine / Hong Kong)."""
    data = [
        ("TSM", "Taiwan Semiconductor (ADR)", "Semi-conducteurs"),
        ("2317.TW", "Hon Hai / Foxconn", "Electronique"),
        ("2454.TW", "MediaTek", "Semi-conducteurs"),
        ("005930.KS", "Samsung Electronics", "Semi-conducteurs / Electronique"),
        ("000660.KS", "SK Hynix", "Semi-conducteurs"),
        ("035420.KS", "Naver", "Internet"),
        ("035720.KS", "Kakao", "Internet"),
        ("9984.T", "SoftBank Group", "Tech / Investissement"),
        ("6758.T", "Sony Group", "Electronique / Divertissement"),
        ("BABA", "Alibaba (ADR)", "E-commerce / Cloud"),
        ("JD", "JD.com (ADR)", "E-commerce"),
        ("BIDU", "Baidu (ADR)", "IA / Internet"),
        ("PDD", "PDD Holdings (ADR)", "E-commerce"),
        ("0700.HK", "Tencent", "Internet / Gaming"),
        ("SE", "Sea Limited (ADR)", "Internet / Gaming"),
    ]
    return pd.DataFrame(data, columns=["Symbol", "Nom", "Groupe"])

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

def load_priority_watchlist() -> pd.DataFrame:
    """Marché cible personnalisé de l'utilisateur : chaîne de valeur semi-conducteurs,
    IA, luxe, énergie/uranium, mines d'or, pharma/biotech, financières/bourses.
    Multi-devises (voir infer_currency, appliqué par titre)."""
    data = [
        # Sept Magnifiques
        ("AAPL", "Apple", "Tech"), ("MSFT", "Microsoft", "Tech"), ("GOOGL", "Alphabet", "Tech"),
        ("AMZN", "Amazon", "Tech"), ("NVDA", "Nvidia", "Semi-conducteurs"), ("META", "Meta Platforms", "Tech"),
        ("TSLA", "Tesla", "Automobile / Energie"),
        # Semi-conducteurs, chaîne de valeur complète
        ("AMD", "AMD", "Semi-conducteurs"), ("INTC", "Intel", "Semi-conducteurs"),
        ("MU", "Micron", "Semi-conducteurs"), ("TSM", "Taiwan Semiconductor (ADR)", "Semi-conducteurs"),
        ("2454.TW", "MediaTek", "Semi-conducteurs"), ("ASX", "ASE Technology (ADR)", "Semi-conducteurs"),
        ("STM", "STMicroelectronics (ADR)", "Semi-conducteurs"),
        ("2317.TW", "Hon Hai Precision (Foxconn)", "Electronique"),
        ("2308.TW", "Delta Electronics", "Electronique"), ("3231.TW", "Wistron", "Electronique"),
        ("BESI.AS", "BE Semiconductor / Besi (équipement)", "Semi-conducteurs"), ("AVGO", "Broadcom", "Semi-conducteurs"),
        ("000660.KS", "SK Hynix", "Semi-conducteurs"), ("005930.KS", "Samsung Electronics", "Semi-conducteurs"),
        ("SOI.PA", "Soitec (semi-conducteurs, wafers)", "Semi-conducteurs"),
        ("ASML", "ASML Holding (équipement lithographie)", "Equipements semi-conducteurs"),
        ("AMAT", "Applied Materials", "Equipements semi-conducteurs"),
        ("LRCX", "Lam Research", "Equipements semi-conducteurs"),
        ("KLAC", "KLA Corp", "Equipements semi-conducteurs"),
        # IA / logiciel / data
        ("BIDU", "Baidu (ADR)", "IA / Internet"), ("WK", "Workiva", "Logiciel"),
        ("SOUN", "SoundHound AI", "IA"), ("PLTR", "Palantir", "IA / Logiciel"),
        # Electronique / optique / matériel
        ("2CRSI.PA", "2CRSI (serveurs IA / datacenter)", "Serveurs / Datacenter"), ("SONY", "Sony Group (ADR)", "Electronique"),
        ("LITE", "Lumentum Holdings", "Optique / Composants"), ("AMBA", "Ambarella", "Semi-conducteurs"),
        ("DELL", "Dell Technologies", "Matériel informatique"),
        # Luxe / conso France
        ("KER.PA", "Kering", "Luxe"), ("MC.PA", "LVMH", "Luxe"), ("RMS.PA", "Hermès", "Luxe"),
        ("CAP.PA", "Capgemini", "Services IT"), ("ORA.PA", "Orange", "Télécoms"),
        # Auto / industrie
        ("STLA", "Stellantis", "Automobile"), ("BYDDY", "BYD (ADR)", "Automobile"),
        ("TM", "Toyota Motors (ADR)", "Automobile"), ("005380.KS", "Hyundai Motor", "Automobile"),
        ("6273.T", "SMC Corp", "Automatisation / Robotique"), ("EMR", "Emerson Electric", "Industrie"),
        ("RR.L", "Rolls Royce", "Aéronautique / Défense"),
        # Energie / uranium / défense nucléaire
        ("TTE.PA", "TotalEnergies", "Energie"), ("CCJ", "Cameco", "Uranium"),
        ("BWXT", "BWX Technologies (composants nucléaires)", "Nucléaire / Défense"),
        # Mines d'or
        ("NEM", "Newmont", "Mines d'or"), ("AEM", "Agnico Eagle Mines", "Mines d'or"),
        ("GOLD", "Barrick Mining (anciennement Barrick Gold)", "Mines d'or"), ("KGC", "Kinross Gold", "Mines d'or"),
        ("6181.HK", "Laopu Gold", "Bijouterie / Or"),
        # Pharma / biotech
        ("207940.KS", "Samsung Biologics", "Biotech"), ("SAN.PA", "Sanofi", "Pharma"),
        ("NVO", "Novo Nordisk (ADR)", "Pharma"), ("LLY", "Eli Lilly", "Pharma"),
        ("IPN.PA", "Ipsen (biopharma)", "Pharma"), ("GNFT.PA", "Genfit (biotech, maladies du foie)", "Biotech"), ("NANO.PA", "Nanobiotix (biotech, oncologie)", "Biotech"),
        # Financières / bourses
        ("JPM", "JPMorgan Chase", "Banque"), ("GS", "Goldman Sachs", "Banque"),
        ("MS", "Morgan Stanley", "Banque"), ("MSCI", "MSCI (indices boursiers, données financières)", "Indices / Data financière"),
        ("ENX.PA", "Euronext (opérateur boursier)", "Bourse"), ("LSEG.L", "LSE Group (opérateur boursier de Londres)", "Bourse"),
        ("G.MI", "Generali (assurance italienne)", "Assurance"), ("ALV.DE", "Allianz (assurance allemande)", "Assurance"),
        ("BLK", "BlackRock", "Gestion d'actifs"), ("MUFG", "Mitsubishi UFJ Financial (ADR)", "Banque"),
        ("V", "Visa", "Paiements"), ("MA", "Mastercard", "Paiements"),
        # Chine
        ("0700.HK", "Tencent Holdings", "Internet / Gaming"), ("BABA", "Alibaba (ADR)", "E-commerce / Cloud"),
        ("300750.SZ", "CATL", "Batteries"),
    ]
    return pd.DataFrame(data, columns=["Symbol", "Nom", "Groupe"])

MARKETS: dict[str, MarketConfig] = {
    "sp500": MarketConfig("sp500", "S&P 500 (États-Unis)", load_sp500, "", "$", "Secteur GICS"),
    "nasdaq100": MarketConfig(
        "nasdaq100", "Nasdaq 100 (États-Unis)", load_nasdaq100, "", "$", "Secteur GICS", is_curated=True,
        note="Composition codée en dur (le scraping Wikipedia s'est révélé trop instable). Confiance "
             "moyenne à haute sur les plus grosses valeurs, plus faible sur la longue traîne : l'indice "
             "compte 100 valeurs avec un renouvellement annuel non négligeable.",
    ),
    "dow30": MarketConfig(
        "dow30", "Dow Jones 30 (États-Unis)", load_dow30, "", "$", "Industrie", is_curated=True,
        note="Composition codée en dur (le scraping Wikipedia s'est révélé trop instable). Indice très "
             "stable (peu de rotations par an), confiance élevée sur cette liste.",
    ),
    "cac40": MarketConfig(
        "cac40", "CAC 40, grandes capitalisations (France)", load_cac40_leaders, "", "€", "Secteur", is_curated=True,
        note="Sélection maison des principales valeurs du CAC 40, liste non exhaustive (le scraping Wikipedia s'est montré peu fiable pour cet indice).",
    ),
    "dax40": MarketConfig(
        "dax40", "DAX 40, grandes capitalisations (Allemagne)", load_dax40_leaders, "", "€", "Secteur", is_curated=True,
        note="Sélection maison des principales valeurs du DAX 40, liste non exhaustive (le scraping Wikipedia s'est montré peu fiable pour cet indice).",
    ),
    "sx5e": MarketConfig(
        "sx5e", "Euro Stoxx 50, SX5E (Zone euro)", load_sx5e_leaders, "", "€", "Secteur", is_curated=True,
        note="Sélection maison proche de la composition du EURO STOXX 50, liste non exhaustive.",
    ),
    "ftse100": MarketConfig("ftse100", "FTSE 100 (Royaume-Uni)", load_ftse100, ".L", "£", "Secteur"),
    "nikkei225": MarketConfig(
        "nikkei225", "Nikkei 225, grandes capitalisations (Japon)", load_nikkei_leaders, "", "¥", "Secteur", is_curated=True,
        note="Sélection maison des principales valeurs du Nikkei 225, liste non exhaustive (le scraping Wikipedia s'est montré peu fiable pour cet indice).",
    ),
    "hangseng": MarketConfig("hangseng", "Hang Seng (Hong Kong)", load_hangseng, ".HK", "HK$", "Secteur"),
    "kospi": MarketConfig(
        "kospi", "Kospi, grandes capitalisations (Corée)", load_kospi_leaders, "", "₩", "Secteur", is_curated=True,
        note="Sélection maison d'une trentaine de grandes valeurs : le KOSPI compte environ 800 sociétés "
             "cotées au total, cette liste n'en couvre qu'une petite partie.",
    ),
    "asia_tech": MarketConfig(
        "asia_tech", "Asie, leaders tech", load_asia_tech_leaders, "", "$", "Pays / Thématique", is_curated=True,
        note="Sélection maison de grandes valeurs technologiques asiatiques (Japon, Corée, Taïwan, Chine).",
    ),
    "tech_ai": MarketConfig(
        "tech_ai", "Tech & IA en vogue", load_tech_trending, "", "$", "Thématique", is_curated=True,
        note="Sélection maison de grandes valeurs tech/IA, pas un indice officiel.",
    ),
    "priority": MarketConfig(
        "priority", "Sélection diversifiée", load_priority_watchlist, "", "mixte", "Thématique", is_curated=True,
        note="Watchlist personnalisée : semi-conducteurs, IA, luxe, énergie/uranium, mines d'or, "
             "pharma/biotech, financières. Multi-devises (affichée par titre).",
    ),
    "crypto": MarketConfig(
        "crypto", "₿ Cryptomonnaies (Top 26)", load_crypto_top, "", "$", "Catégorie", is_curated=True,
        note="Sélection maison des cryptos majeures, vérifiez que chaque ticker est bien coté sur Yahoo Finance.",
    ),
    "custom": MarketConfig(
        "custom", "Marché personnalisé", None, "", "$", "Groupe", is_curated=True,
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

def _ramp_score(value, lo, hi, max_pts):
    """max_pts si value<=lo, 0 si value>=hi, interpolation linéaire entre les deux.
    Évite les paliers brutaux (ex : RSI 39,9 valant 15 pts de plus que RSI 40,1)."""
    if value <= lo:
        return max_pts
    if value >= hi:
        return 0.0
    return max_pts * (hi - value) / (hi - lo)

def compute_score(rsi, price, boll_low, boll_mid, vol_ratio, macd_hist_prev, macd_hist_last,
                   pct_from_low, var_1j, var_5j) -> int:
    """Score d'opportunité 0-100. Combine survente (RSI), extension des Bandes
    de Bollinger, momentum baissier récent, anomalie de volume, retournement
    MACD naissant et proximité du plus bas sur la période.

    Deux choix structurants par rapport à une simple somme de seuils booléens :
    - Les composantes RSI et "plus bas de la période" sont en rampe continue
      plutôt qu'à paliers : un RSI de 40,1 n'attribue plus 15 points de moins
      qu'un RSI de 39,9.
    - Un momentum de baisse récent (Var. 1J / 5J) est explicitement récompensé,
      et un rebond récent réduit le score même si RSI/Bollinger restent
      techniquement en zone de survente (ces indicateurs réagissent avec
      retard et peuvent laisser un titre "collé" en haut du classement
      plusieurs jours après que l'opportunité s'est déjà estompée).
    """
    score = 0.0

    # RSI (0-25 pts) : rampe continue de 25 pts à RSI<=15 jusqu'à 0 pt à RSI>=50.
    if pd.notna(rsi):
        score += _ramp_score(rsi, lo=15, hi=50, max_pts=25)

    # Bandes de Bollinger (0-20 pts) : proportionnel à la profondeur de pénétration
    # sous la bande basse (normalisée par la largeur du canal), pas un simple booléen.
    if pd.notna(boll_low) and pd.notna(boll_mid) and price <= boll_low and boll_mid > boll_low:
        depth = (boll_low - price) / (boll_mid - boll_low)
        score += min(20, depth * 40)

    # Momentum baissier récent (0-20 pts) : le signal le plus direct d'une opportunité
    # fraîche. Une variation positive ne retire rien ici (elle rapporte 0), mais réduit
    # aussi la composante volume ci-dessous.
    mom = 0.0
    if pd.notna(var_1j):
        mom += max(0.0, min(10.0, -var_1j * 2.5))
    if pd.notna(var_5j):
        mom += max(0.0, min(10.0, -var_5j * 1.0))
    score += mom

    # Volume anormal (0-15 pts) : un volume élevé n'est un signal d'opportunité que sur
    # une séance stable ou baissière (capitulation/distribution). Sur une forte hausse,
    # un volume élevé est un signal haussier, pas une occasion de repli.
    if pd.notna(vol_ratio) and vol_ratio > 1.2:
        vol_pts = min(15.0, (vol_ratio - 1.0) * 15.0)
        if pd.notna(var_1j) and var_1j > 0.5:
            vol_pts *= 0.3
        score += vol_pts

    # Retournement MACD naissant (0-10 pts)
    if pd.notna(macd_hist_prev) and pd.notna(macd_hist_last) and macd_hist_prev <= 0 and macd_hist_last > 0:
        score += 10.0

    # Proximité du plus bas sur la période (0-10 pts) : rampe continue.
    if pd.notna(pct_from_low):
        score += max(0.0, 10.0 - pct_from_low / 2.0)

    return int(round(min(max(score, 0.0), 100.0)))

# ══════════════════════════════════════════════════════════════════════════
# 4. PIPELINE DE RÉCUPÉRATION & D'ANALYSE
# ══════════════════════════════════════════════════════════════════════════

def _extract_frame(data: pd.DataFrame, symbol: str, n_tickers: int) -> pd.DataFrame:
    """yfinance (avec group_by='ticker') peut renvoyer un MultiIndex même
    pour un seul ticker : on se base donc sur la structure réelle des
    colonnes plutôt que sur le nombre de tickers demandés."""
    if isinstance(data.columns, pd.MultiIndex):
        if symbol not in data.columns.get_level_values(0):
            return pd.DataFrame()
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
        return pd.DataFrame(), f"{e}", []

    if data is None or data.empty:
        return pd.DataFrame(), "Aucune donnée reçue (tickers invalides ou délistés, limite de requêtes Yahoo Finance, ou accès réseau restreint).", []

    rows = []
    skipped = []  # (symbol, raison) : rendu visible dans l'UI plutôt que silencieusement perdu
    for symbol in symbols:
        try:
            df_s = _extract_frame(data, symbol, n)
            if len(df_s) == 0:
                skipped.append((symbol, "aucune donnée renvoyée par Yahoo Finance"))
                continue
            if "Close" not in df_s.columns:
                skipped.append((symbol, "données incomplètes (colonne Close absente)"))
                continue
            if len(df_s) < 30:
                skipped.append((symbol, f"historique trop court ({len(df_s)} séances, 30 minimum)"))
                continue

            close = df_s["Close"]
            last_p, prev_p = float(close.iloc[-1]), float(close.iloc[-2])
            var_day = (last_p - prev_p) / prev_p * 100
            var_5d = (last_p - float(close.iloc[-5])) / float(close.iloc[-5]) * 100 if len(close) >= 5 else np.nan

            ind = compute_indicators(df_s)
            rsi = ind["rsi"].iloc[-1]
            boll_low = ind["boll_low"].iloc[-1]
            boll_mid = ind["sma20"].iloc[-1]
            vol_ratio = ind["vol_ratio"].iloc[-1]
            macd_hist = ind["macd_hist"]
            macd_prev = macd_hist.iloc[-2] if len(macd_hist) >= 2 else np.nan
            macd_last = macd_hist.iloc[-1]

            period_low = float(close.min())
            pct_from_low = (last_p - period_low) / period_low * 100 if period_low else np.nan

            score = compute_score(rsi, last_p, boll_low, boll_mid, vol_ratio, macd_prev, macd_last,
                                   pct_from_low, var_day, var_5d)

            rows.append({
                "Ticker": symbol,
                "Nom": shorten_name(names_map.get(symbol, symbol)),
                "Groupe": shorten_sector(groups_map.get(symbol, "N/A")),
                "Prix": round(last_p, 2),
                "Var. 1J (%)": round(var_day, 2),
                "Var. 5J (%)": round(var_5d, 2) if pd.notna(var_5d) else None,
                "RSI (14)": round(float(rsi), 1) if pd.notna(rsi) else None,
                "Sous Bollinger": bool(pd.notna(boll_low) and last_p <= boll_low),
                "Ratio Vol.": round(float(vol_ratio), 2) if pd.notna(vol_ratio) else None,
                "% vs Bas (période)": round(pct_from_low, 1) if pd.notna(pct_from_low) else None,
                "MACD haussier": bool(pd.notna(macd_prev) and pd.notna(macd_last) and macd_prev <= 0 and macd_last > 0),
                "Devise": infer_currency(symbol),
                "Score Opp.": score,
                "_history": df_s,
            })
        except Exception as e:
            skipped.append((symbol, f"erreur inattendue : {e}"))
            continue

    return pd.DataFrame(rows), None, skipped

# ══════════════════════════════════════════════════════════════════════════
# 4B. CARTES D'OPPORTUNITE & JAUGE DE SCORE (SVG)
# ══════════════════════════════════════════════════════════════════════════

def render_gauge_svg(score: int) -> str:
    """Jauge de score en demi-cercle, en SVG pur plutôt qu'en Plotly.

    Le SVG est dimensionné en pourcentage (width="100%") avec un viewBox
    fixe : le texte reste donc parfaitement centré quelle que soit la
    largeur réelle du conteneur (1, 2 ou 3 cartes affichées, mobile ou
    grand écran), ce qu'un graphique Plotly redimensionné ne garantit pas.
    Rendu sur une seule ligne pour éviter tout risque d'interruption du
    bloc HTML par le moteur Markdown (voir render_hero_icon_svg).
    """
    radius = 80
    circumference = math.pi * radius
    pct = max(0, min(100, score)) / 100
    offset = circumference * (1 - pct)
    if score >= 70:
        color = "#22C55E"
    elif score >= 40:
        color = "#F5A623"
    else:
        color = "#F0466E"
    return (
        '<div class="gauge-wrap">'
        f'<svg viewBox="0 0 200 114" class="gauge-svg" role="img" aria-label="Score {score} sur 100">'
        '<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#2E3A56" '
        'stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" '
        f'stroke-linecap="round" stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"/>'
        f'<text x="100" y="84" text-anchor="middle" class="gauge-value">{score}</text>'
        '<text x="100" y="104" text-anchor="middle" class="gauge-suffix">/ 100</text>'
        '</svg></div>'
    )

def render_opportunity_card(row: pd.Series, currency: str, rank_label: str) -> None:
    """Affiche une carte d'opportunité complète (texte + jauge) comme un seul
    bloc HTML auto-porté (bordure/fond/survol inclus) : à appeler directement
    dans une colonne, sans `st.container(border=True)` autour."""
    var_cls = "badge-buy" if row["Var. 1J (%)"] >= 0 else "badge-down"
    badges = f'<span class="badge {var_cls}">Var 1J {row["Var. 1J (%)"]:+.2f}%</span>'
    if row["Sous Bollinger"]:
        badges += '<span class="badge badge-warn">Sous Bollinger</span>'
    if row["Ratio Vol."] and row["Ratio Vol."] > 1.2:
        badges += f'<span class="badge badge-neutral">Vol. {row["Ratio Vol."]:.1f}x</span>'
    if row["MACD haussier"]:
        badges += '<span class="badge badge-buy">MACD haussier</span>'
    pe = fetch_pe_ratio(row["Ticker"])
    if pe is not None:
        badges += f'<span class="badge badge-neutral">P/E {pe}</span>'

    gauge_svg = render_gauge_svg(int(row["Score Opp."]))
    st.markdown(
        f'<div class="opp-card-shell">'
        f'<div class="opp-rank">{rank_label}</div>'
        f'<div class="opp-name">{row["Nom"]}</div>'
        f'<div class="opp-ticker">{row["Ticker"]}</div>'
        f'<div class="opp-price">{currency}{row["Prix"]:.2f}</div>'
        f'<div>{badges}</div>'
        f'{gauge_svg}'
        f'</div>',
        unsafe_allow_html=True,
    )

def infer_currency(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".PA") or t.endswith(".DE") or t.endswith(".MC") or t.endswith(".MI") or t.endswith(".AS"):
        return "€"
    if t.endswith(".L"):
        return "£"
    if t.endswith(".HK"):
        return "HK$"
    if t.endswith(".T"):
        return "¥"
    if t.endswith(".KS") or t.endswith(".KQ"):
        return "₩"
    if t.endswith(".TW"):
        return "NT$"
    if t.endswith(".SZ") or t.endswith(".SS"):
        return "¥"
    if t.endswith(".CO"):
        return "kr"
    return "$"

# ══════════════════════════════════════════════════════════════════════════
# 4C. NOMS ET SECTEURS COURTS (jamais tronqués mid-mot dans les tableaux)
# ══════════════════════════════════════════════════════════════════════════

_LEGAL_SUFFIX_RE = re.compile(
    r"[,]?\s*(inc\.?|corp(oration)?\.?|co\.?|company|plc|ag|se|n\.?v\.?|s\.?a\.?|"
    r"ltd\.?|limited|llc|group|holdings?|kgaa|asa|spa)$",
    flags=re.IGNORECASE,
)

_NAME_OVERRIDES = {
    "super micro computer": "Supermicro",
    "mitsubishi ufj financial group": "Mitsubishi UFJ",
    "deutsche post dhl group": "DHL Group",
    "nippon telegraph and telephone": "NTT",
    "mercedes-benz group": "Mercedes-Benz",
    "fast retailing (uniqlo)": "Uniqlo",
    "hon hai / foxconn": "Foxconn",
    "sea limited (adr)": "Sea Limited",
    "pdd holdings (adr)": "PDD Holdings",
}

def shorten_name(name: str, max_len: int = 24) -> str:
    """Nom court et compréhensible plutôt que tronqué au milieu par
    l'interface : on retire les suffixes juridiques (Inc., Group, SE...)
    puis, si besoin, on coupe à la dernière frontière de mot entière."""
    n = str(name).strip()
    if not n:
        return n
    override = _NAME_OVERRIDES.get(n.lower())
    if override:
        return override
    prev = None
    while prev != n:
        prev = n
        n = _LEGAL_SUFFIX_RE.sub("", n).strip()
    if len(n) > max_len:
        words = n.split(" ")
        short = words[0]
        for w in words[1:]:
            if len(short) + 1 + len(w) > max_len:
                break
            short = f"{short} {w}"
        n = short
    return n.strip() or str(name).strip()


_SECTOR_SHORT = {
    "consommation": "Conso",
    "information technology": "Tech. Info.",
    "consumer discretionary": "Conso. discr.",
    "consumer staples": "Conso. base",
    "communication services": "Communication",
    "health care": "Santé",
    "financials": "Finance",
    "industrials": "Industrie",
    "real estate": "Immobilier",
    "materials": "Matériaux",
    "utilities": "Services publics",
    "energy": "Énergie",
    "semi-conducteurs / ia": "Semi-cond. / IA",
    "semi-conducteurs / electronique": "Semi-cond. / Électro.",
    "cloud / saas": "Cloud / SaaS",
    "cloud / ia": "Cloud / IA",
    "tech / investissement": "Tech / Invest.",
}

def shorten_sector(sector: str, max_len: int = 18) -> str:
    s = str(sector).strip()
    if not s:
        return s
    override = _SECTOR_SHORT.get(s.lower())
    if override:
        return override
    if len(s) > max_len:
        words = s.split(" ")
        short = words[0]
        for w in words[1:]:
            if len(short) + 1 + len(w) > max_len:
                break
            short = f"{short} {w}"
        return short.strip()
    return s

# ══════════════════════════════════════════════════════════════════════════
# 4D. RECHERCHE LIBRE, TOUS MARCHÉS (résolution nom -> ticker)
# ══════════════════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    """Normalise une chaîne pour la comparaison : minuscules, sans accents,
    sans ponctuation ni espaces."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _build_aliases() -> dict:
    """Table de correspondance nom d'entreprise -> (ticker, nom affiché).

    Volontairement limitée aux actions ordinaires (aucun produit dérivé,
    ETF à levier, warrant ou option) sur des sociétés dont le ticker
    principal est bien établi, afin de limiter le risque d'erreur. Elle est
    complétée par les listes déjà curées de l'application (Tech & IA,
    cryptomonnaies, Kospi, Asie tech).
    """
    raw = {
        # Méga-capitalisations US
        "apple": ("AAPL", "Apple"), "microsoft": ("MSFT", "Microsoft"),
        "google": ("GOOGL", "Alphabet"), "alphabet": ("GOOGL", "Alphabet"),
        "amazon": ("AMZN", "Amazon"), "nvidia": ("NVDA", "Nvidia"),
        "meta": ("META", "Meta Platforms"), "facebook": ("META", "Meta Platforms"),
        "tesla": ("TSLA", "Tesla"), "amd": ("AMD", "AMD"),
        "broadcom": ("AVGO", "Broadcom"), "oracle": ("ORCL", "Oracle"),
        "salesforce": ("CRM", "Salesforce"), "adobe": ("ADBE", "Adobe"),
        "netflix": ("NFLX", "Netflix"), "palantir": ("PLTR", "Palantir"),
        "intel": ("INTC", "Intel"), "ibm": ("IBM", "IBM"),
        "qualcomm": ("QCOM", "Qualcomm"), "servicenow": ("NOW", "ServiceNow"),
        "micron": ("MU", "Micron"), "supermicro": ("SMCI", "Super Micro Computer"),
        # CAC 40
        "lvmh": ("MC.PA", "LVMH"), "loreal": ("OR.PA", "L'Oréal"),
        "totalenergies": ("TTE", "TotalEnergies"), "total": ("TTE", "TotalEnergies"),
        "sanofi": ("SAN.PA", "Sanofi"), "bnpparibas": ("BNP.PA", "BNP Paribas"),
        "axa": ("CS.PA", "AXA"), "airliquide": ("AI.PA", "Air Liquide"),
        "airbus": ("AIR.PA", "Airbus"), "danone": ("BN.PA", "Danone"),
        "renault": ("RNO.PA", "Renault"), "carrefour": ("CA.PA", "Carrefour"),
        "vinci": ("DG.PA", "Vinci"), "kering": ("KER.PA", "Kering"),
        "schneiderelectric": ("SU.PA", "Schneider Electric"),
        "saintgobain": ("SGO.PA", "Saint-Gobain"),
        "essilorluxottica": ("EL.PA", "EssilorLuxottica"), "hermes": ("RMS.PA", "Hermès"),
        "societegenerale": ("GLE.PA", "Société Générale"),
        "creditagricole": ("ACA.PA", "Crédit Agricole"), "bouygues": ("EN.PA", "Bouygues"),
        "michelin": ("ML.PA", "Michelin"), "thales": ("HO.PA", "Thales"),
        "safran": ("SAF.PA", "Safran"), "publicis": ("PUB.PA", "Publicis"),
        "legrand": ("LR.PA", "Legrand"), "capgemini": ("CAP.PA", "Capgemini"),
        "dassaultsystemes": ("DSY.PA", "Dassault Systèmes"),
        "pernodricard": ("RI.PA", "Pernod Ricard"), "veolia": ("VIE.PA", "Veolia"),
        "engie": ("ENGI.PA", "Engie"), "orange": ("ORA.PA", "Orange"),
        "stmicroelectronics": ("STM", "STMicroelectronics"),
        # DAX 40
        "sap": ("SAP", "SAP"), "siemens": ("SIE.DE", "Siemens"),
        "volkswagen": ("VOW3.DE", "Volkswagen"), "bmw": ("BMW.DE", "BMW"),
        "mercedesbenz": ("MBG.DE", "Mercedes-Benz Group"), "allianz": ("ALV.DE", "Allianz"),
        "adidas": ("ADS.DE", "Adidas"), "deutschebank": ("DBK.DE", "Deutsche Bank"),
        "basf": ("BAS.DE", "BASF"), "bayer": ("BAYN.DE", "Bayer"),
        "infineon": ("IFX.DE", "Infineon"),
        # Grandes ADR internationales bien établies
        "shell": ("SHEL", "Shell"), "hsbc": ("HSBC", "HSBC"),
        "astrazeneca": ("AZN", "AstraZeneca"), "bp": ("BP", "BP"),
        "unilever": ("UL", "Unilever"), "gsk": ("GSK", "GSK"),
    }
    aliases = dict(raw)
    for loader in (load_tech_trending, load_crypto_top, load_kospi_leaders, load_asia_tech_leaders,
                   load_cac40_leaders, load_dax40_leaders, load_nikkei_leaders, load_sx5e_leaders):
        try:
            for _, r in loader().iterrows():
                key = _norm(r["Nom"])
                if key and key not in aliases:
                    aliases[key] = (r["Symbol"], r["Nom"])
        except Exception:
            continue
    return {k: v for k, v in aliases.items() if k}

ALIASES = _build_aliases()

def resolve_query_to_ticker(query: str, universe_df: pd.DataFrame, results_df: pd.DataFrame):
    """Résout une saisie libre (ticker ou nom d'entreprise) en un ticker
    Yahoo Finance, indépendamment du marché actuellement sélectionné.

    Ordre de résolution :
      1. Table d'alias curée (cross-marché) ;
      2. Correspondance exacte ou partielle dans l'univers déjà chargé
         (marché actuellement sélectionné) ;
      3. Correspondance partielle dans la table d'alias ;
      4. Si la saisie ressemble déjà à un ticker valide (lettres/chiffres/
         '.', '-', '^'), on la tente telle quelle.

    Retourne (ticker, nom_affiché) ou (None, None) si rien n'est résolu.
    """
    q = query.strip()
    if not q:
        return None, None
    nq = _norm(q)

    if nq in ALIASES:
        return ALIASES[nq]

    for df, ticker_col in ((results_df, "Ticker"), (universe_df, "Symbol")):
        if df is None or len(df) == 0 or "Nom" not in df.columns or ticker_col not in df.columns:
            continue
        norm_names = df["Nom"].astype(str).map(_norm)
        exact = df[norm_names == nq]
        if len(exact):
            r = exact.iloc[0]
            return r[ticker_col], r["Nom"]
        ticker_exact = df[df[ticker_col].astype(str).str.lower() == q.lower()]
        if len(ticker_exact):
            r = ticker_exact.iloc[0]
            return r[ticker_col], r["Nom"]
        contains = df[norm_names.str.contains(nq, na=False)] if nq else df.iloc[0:0]
        if len(contains):
            r = contains.iloc[0]
            return r[ticker_col], r["Nom"]

    for key, (ticker, name) in ALIASES.items():
        if nq and (nq in key or key in nq):
            return ticker, name

    if re.fullmatch(r"[\^A-Za-z0-9.\-]{1,12}", q):
        return q.upper(), q.upper()

    return None, None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_single_ticker(ticker: str, display_name: str):
    """Récupère et analyse un seul ticker, indépendamment du marché
    sélectionné dans la barre latérale. Réutilise le même pipeline
    d'indicateurs/scoring que l'analyse principale."""
    df_result, err, _skipped = fetch_and_analyze("search", [ticker], {ticker: display_name}, {ticker: "Recherche"})
    if err or df_result.empty:
        return None
    return df_result.iloc[0]

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_pe_ratio(ticker: str):
    """P/E (cours/bénéfices) à titre informatif uniquement, jamais intégré au
    score (voir Méthodologie). Volontairement PAS appelé pour tout un marché :
    contrairement à yf.download (un seul appel groupé), .info fait une requête
    réseau par titre, ce qui rendrait le scan d'un marché de plusieurs
    centaines de valeurs beaucoup trop lent et sujet aux limites de débit de
    Yahoo Finance. Utilisé uniquement pour les quelques titres réellement mis
    en avant à l'écran (cartes spotlight, recherche, onglet graphique)."""
    try:
        info = yf.Ticker(ticker).get_info()
        pe = info.get("trailingPE")
        return round(pe, 1) if pe and pe > 0 else None
    except Exception:
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_chart_history(ticker: str) -> pd.DataFrame:
    """Historique dédié pour l'onglet graphique (5 ans, quotidien), séparé
    du fetch d'analyse global (limité à 1 an pour ne pas ralentir le
    screening de tout un marché). Permet les plages longues (1A, 5A, Tout)."""
    try:
        data = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=True)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def slice_by_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    """Découpe l'historique selon une plage classique (façon Yahoo Finance /
    Google Finance). Si le titre est coté depuis moins longtemps que la plage
    demandée (ex : IPO récente + plage "5 ans"), le filtre par date ne peut
    rien exclure et renvoie simplement tout l'historique disponible."""
    if df.empty:
        return df
    last_date = df.index.max()
    if range_key == "1M":
        return df[df.index >= last_date - pd.Timedelta(days=31)]
    if range_key == "6M":
        return df[df.index >= last_date - pd.Timedelta(days=183)]
    if range_key == "YTD":
        return df[df.index >= pd.Timestamp(year=last_date.year, month=1, day=1)]
    if range_key == "1A":
        return df[df.index >= last_date - pd.Timedelta(days=366)]
    if range_key == "5A":
        return df[df.index >= last_date - pd.Timedelta(days=5 * 366)]
    return df  # "Tout"

# ══════════════════════════════════════════════════════════════════════════
# 5. SIDEBAR : SELECTION DU MARCHE ET FILTRES
# ══════════════════════════════════════════════════════════════════════════
st.sidebar.markdown('<div class="sidebar-title">Marché et univers</div>', unsafe_allow_html=True)
market_key = st.sidebar.selectbox(
    "Marché à analyser", list(MARKETS.keys()), format_func=lambda k: MARKETS[k].label,
)
market = MARKETS[market_key]
if market.note:
    st.sidebar.caption(f"ℹ️ {market.note}")

if market.key == "custom":
    st.session_state.setdefault("custom_watchlist", [])

    with st.sidebar.form("add_ticker_form", clear_on_submit=True, border=False):
        query = st.text_input("Ajouter une entreprise (nom ou ticker)", placeholder="ex : Nvidia, MC.PA, BTC-USD")
        add_clicked = st.form_submit_button("Ajouter à mon marché", width="stretch")
    if add_clicked and query.strip():
        resolved_ticker, resolved_name = resolve_query_to_ticker(query, pd.DataFrame(), pd.DataFrame())
        if resolved_ticker is None:
            st.sidebar.warning(f"« {query} » non reconnu. Essayez un ticker exact (ex : NVDA, MC.PA).")
        elif resolved_ticker in {r["Symbol"] for r in st.session_state["custom_watchlist"]}:
            st.sidebar.info(f"{resolved_ticker} est déjà dans votre marché.")
        else:
            st.session_state["custom_watchlist"].append({"Symbol": resolved_ticker, "Nom": resolved_name})

    st.sidebar.file_uploader(
        "Ou recharger un marché déjà enregistré (CSV)", type=["csv"], key="custom_csv_upload",
        help="Importe une liste Symbol/Nom exportée précédemment. L'analyse sera recalculée avec "
             "les données de marché actuelles, pas celles du jour de l'export.",
    )
    if st.session_state.get("custom_csv_upload") is not None:
        try:
            imported = pd.read_csv(st.session_state["custom_csv_upload"])
            imported.columns = [c.strip() for c in imported.columns]
            if "Symbol" not in imported.columns:
                st.sidebar.error("Le fichier CSV doit contenir au moins une colonne 'Symbol'.")
            else:
                if "Nom" not in imported.columns:
                    imported["Nom"] = imported["Symbol"]
                st.session_state["custom_watchlist"] = (
                    imported[["Symbol", "Nom"]].dropna(subset=["Symbol"]).drop_duplicates("Symbol").to_dict("records")
                )
                st.sidebar.success(f"{len(st.session_state['custom_watchlist'])} titre(s) rechargé(s) depuis le CSV.")
        except Exception as e:
            st.sidebar.error(f"Fichier CSV illisible : {e}")

    if st.session_state["custom_watchlist"]:
        watchlist_df = pd.DataFrame(st.session_state["custom_watchlist"])
        edited_df = st.sidebar.data_editor(
            watchlist_df, hide_index=True, width="stretch", num_rows="dynamic", key="custom_watchlist_editor",
        )
        st.session_state["custom_watchlist"] = edited_df.dropna(subset=["Symbol"]).to_dict("records")

        csv_bytes = pd.DataFrame(st.session_state["custom_watchlist"]).to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            "Enregistrer ce marché (CSV)", data=csv_bytes,
            file_name=f"mon_marche_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv",
            mime="text/csv", width="stretch",
        )
    else:
        st.sidebar.caption("Aucun titre pour l'instant : recherchez une entreprise ci-dessus ou importez un CSV.")

    tickers_raw = [r["Symbol"] for r in st.session_state["custom_watchlist"]]
    names_raw = {r["Symbol"]: r["Nom"] for r in st.session_state["custom_watchlist"]}
    universe_df = pd.DataFrame({
        "Symbol": tickers_raw,
        "Nom": [names_raw[t] for t in tickers_raw],
        "Groupe": "Personnalisé",
    })
else:
    try:
        with st.spinner(f"Chargement de la composition : {market.label}..."):
            universe_df = market.loader()
    except Exception as e:
        st.sidebar.error(f"Impossible de charger la composition du marché : {e}")
        universe_df = pd.DataFrame(columns=["Symbol", "Nom", "Groupe"])

st.sidebar.caption(f"{len(universe_df)} titres dans l'univers sélectionné")
st.sidebar.markdown("---")

st.sidebar.markdown('<div class="sidebar-title">Profil et filtres</div>', unsafe_allow_html=True)

PRESETS = {
    "Conservateur": {"score": 60, "rsi": 30},
    "Modéré": {"score": 40, "rsi": 35},
    "Agressif": {"score": 25, "rsi": 45},
}

def _apply_preset():
    p = PRESETS.get(st.session_state.get("preset_choice"))
    if p:
        st.session_state["min_score"] = p["score"]
        st.session_state["rsi_max"] = p["rsi"]

def _mark_custom():
    st.session_state["preset_choice"] = "Personnalisé"

st.session_state.setdefault("min_score", 40)
st.session_state.setdefault("rsi_max", 35)
st.session_state.setdefault("preset_choice", "Modéré")
st.sidebar.radio(
    "Profil de risque", list(PRESETS.keys()) + ["Personnalisé"],
    key="preset_choice", on_change=_apply_preset,
)

with st.sidebar.expander("Critères de sélection", expanded=True):
    group_options = ["Tous"] + sorted({shorten_sector(g) for g in universe_df["Groupe"].dropna()}) if len(universe_df) else ["Tous"]
    selected_group = st.selectbox(market.group_label, group_options)
    min_score = st.slider("Score Opportunité Min.", 1, 100, step=5, key="min_score", on_change=_mark_custom)
    rsi_max = st.slider("RSI Max (zone de survente)", 10, 50, key="rsi_max", on_change=_mark_custom)
    st.caption(
        "Score et RSI sont deux filtres indépendants (ET logique) : baisser le score minimum "
        "n'affiche rien tant que le RSI dépasse ce seuil. Si un marché reste vide, montez ce curseur."
    )
    volume_filter = st.checkbox("Volume ≥ moyenne 20 jours (≥ 1.0x)", value=False)
    search_query = st.text_input("Filtrer le tableau affiché", "")
    st.caption("Pour chercher un titre hors du marché sélectionné, utilisez la recherche libre en haut de page.")

with st.sidebar.expander("Affichage"):
    max_rows = st.select_slider("Résultats affichés (max.)", options=[10, 20, 50, 100, "Tous"], value=50)
    sort_col = st.selectbox("Trier par", ["Score Opp.", "RSI (14)", "Var. 1J (%)", "Ratio Vol."], index=0)

st.sidebar.markdown("---")
if st.sidebar.button("Rafraîchir les données", width="stretch"):
    fetch_and_analyze.clear()
    st.rerun()
st.sidebar.caption(f"Dernière analyse : {dt.datetime.now().strftime('%d/%m/%Y %H:%M')}")

# ══════════════════════════════════════════════════════════════════════════
# 6. RÉCUPÉRATION DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════
if len(universe_df) == 0:
    st.markdown(
        '<div class="hero"><div class="hero-badge">En attente</div>'
        f'<div class="hero-title-row">{render_hero_icon_svg()}'
        "<h1 class=\"gradient-text\">Screener d'Opportunités, Court Terme et Rebond</h1></div>"
        '<p>Sélectionnez un marché ou saisissez des tickers personnalisés dans la barre latérale pour démarrer l\'analyse.</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()

symbols = [_ensure_suffix(str(s), market.suffix) for s in universe_df["Symbol"]]
names_map = dict(zip(symbols, universe_df["Nom"]))
groups_map = dict(zip(symbols, universe_df["Groupe"]))

with st.spinner(f"Analyse de {len(symbols)} titres : {market.label}..."):
    results_df, fetch_error, skipped_symbols = fetch_and_analyze(market.key, symbols, names_map, groups_map)

if fetch_error:
    st.error(f"Erreur lors de la récupération des données : {fetch_error}")
    st.stop()
if results_df.empty:
    st.warning("Aucune donnée exploitable n'a pu être calculée pour cet univers. Réessayez ou changez de marché.")
    st.stop()
if skipped_symbols:
    with st.expander(f"{len(skipped_symbols)} titre(s) ignoré(s) sur {len(symbols)} (données indisponibles)"):
        for sym, reason in skipped_symbols:
            st.markdown(f"- **{sym}** : {reason}")

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
      <div class="hero-badge">Marché actif : {market.label}</div>
      <div class="hero-title-row">
        {render_hero_icon_svg()}
        <h1 class="gradient-text">Screener d'Opportunités, Court Terme et Rebond</h1>
      </div>
      <p>Détection automatisée des titres en survente, avec volume anormal et signaux techniques
      de retournement (RSI, Bandes de Bollinger, MACD), adaptable à tous les marchés.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════
# 10. KPIs
# ══════════════════════════════════════════════════════════════════════════
kpis = [
    ("Univers analysé", str(len(results_df))),
    ("Opportunités détectées", str(len(filtered))),
    ("RSI moyen (sélection)", f"{filtered['RSI (14)'].mean():.1f}" if len(filtered) else "N/A"),
    ("Variation moy. 1J", f"{filtered['Var. 1J (%)'].mean():+.2f}%" if len(filtered) else "N/A"),
    ("Score moyen", f"{filtered['Score Opp.'].mean():.0f} / 100" if len(filtered) else "N/A"),
]
kpi_html = "".join(
    f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'
    for label, value in kpis
)
st.markdown(f'<div class="kpi-grid">{kpi_html}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# 11. SPOTLIGHT : TOP 3 OPPORTUNITES
# ══════════════════════════════════════════════════════════════════════════
if len(filtered) > 0:
    st.markdown('#### <span class="gradient-text">Meilleures opportunités</span>', unsafe_allow_html=True)
    top3 = filtered.head(3)
    cols = st.columns(len(top3))
    for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
        with col:
            render_opportunity_card(row, row["Devise"], f"Opportunité n°{i + 1}")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# 11B. RECHERCHE LIBRE, TOUS MARCHÉS
# ══════════════════════════════════════════════════════════════════════════
st.markdown('#### <span class="gradient-text">Rechercher un titre, sur tous les marchés</span>', unsafe_allow_html=True)
st.markdown(
    '<div class="panel-sub">Ticker (ex: MC.PA, ^FCHI, AAPL) ou nom d\'entreprise (ex: Orange, Sanofi, Apple). '
    'Indépendant du marché sélectionné ci-contre.</div>',
    unsafe_allow_html=True,
)
col_search, col_btn = st.columns([4, 1])
with col_search:
    free_query = st.text_input(
        "Ticker ou nom d'entreprise", "", key="free_search",
        label_visibility="collapsed", placeholder="Ticker ou nom d'entreprise, ex: Orange, Sanofi, MC.PA, ^FCHI",
    )
with col_btn:
    search_clicked = st.button("Rechercher", width="stretch")

if search_clicked and free_query.strip():
    resolved_ticker, resolved_name = resolve_query_to_ticker(free_query, universe_df, results_df)
    if resolved_ticker is None:
        st.warning(
            f"Aucun titre trouvé pour « {free_query} ». Essayez le ticker exact au format Yahoo Finance "
            "(ex: MC.PA, ^FCHI) ou le nom d'une entreprise connue (ex: Sanofi, Orange, Apple)."
        )
    else:
        with st.spinner(f"Recherche de {resolved_ticker}..."):
            search_row = fetch_single_ticker(resolved_ticker, resolved_name or resolved_ticker)
        if search_row is None:
            st.warning(
                f"Impossible de récupérer les données pour {resolved_ticker}. "
                "Le ticker est peut-être invalide, délisté, ou l'accès réseau est temporairement limité."
            )
        else:
            search_col = st.columns([1, 1, 1])[0]
            with search_col:
                render_opportunity_card(search_row, infer_currency(resolved_ticker), "Résultat de recherche")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# 12. ONGLETS PRINCIPAUX
# ══════════════════════════════════════════════════════════════════════════
tab_table, tab_heat, tab_chart, tab_about = st.tabs(
    ["Sélection", "Cartographie sectorielle", "Analyse graphique", "Méthodologie"]
)

# ---- Onglet Sélection -------------------------------------------------
with tab_table:
    if len(filtered) == 0:
        st.warning("Aucune action ne correspond aux critères actuels. Assouplissez les filtres dans la barre latérale.")
    else:
        display_df = filtered.drop(columns=["_history"]).copy()
        display_df["Prix"] = display_df.apply(lambda r: f"{r['Devise']}{r['Prix']:.2f}", axis=1)
        display_df = display_df.drop(columns=["Devise"])
        display_df["Lien"] = "https://finance.yahoo.com/quote/" + display_df["Ticker"]
        display_df["Sous Bollinger"] = display_df["Sous Bollinger"].map({True: "Oui", False: "Non"})
        display_df["MACD haussier"] = display_df["MACD haussier"].map({True: "Oui", False: "Non"})

        with st.container(border=True):
            st.markdown('<div class="panel-title">Tableau des opportunités</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="panel-sub">{len(display_df)} titre(s) correspondant aux critères, triés par {sort_col.lower()}</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Score Opp.": st.column_config.ProgressColumn(format="%d/100", min_value=0, max_value=100),
                    "Var. 1J (%)": st.column_config.NumberColumn(format="%.2f %%"),
                    "Var. 5J (%)": st.column_config.NumberColumn(format="%.2f %%"),
                    "Ratio Vol.": st.column_config.NumberColumn(format="%.2f x"),
                    "% vs Bas (période)": st.column_config.NumberColumn(format="%.1f %%"),
                    "Lien": st.column_config.LinkColumn("Fiche", display_text="Voir"),
                },
            )

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Exporter en CSV",
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
                    "Exporter en Excel",
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
        fig.update_traces(texttemplate="<b>%{label}</b><br>%{color:+.2f}%", textposition="middle center",
                           textfont_size=13, marker_line_color="#0A0E1A", marker_line_width=2)
        fig.update_layout(
            height=620, margin=dict(l=6, r=6, t=34, b=6),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E7ECF5"},
        )
        with st.container(border=True):
            st.markdown('<div class="panel-title">Cartographie du marché</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-sub">Taille = poids égal par titre. Couleur = variation du jour '
                '(rouge = baisse, vert = hausse).</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, width="stretch", key="treemap")

# ---- Onglet Analyse graphique ------------------------------------------
with tab_chart:
    if len(results_df) == 0:
        st.info("Pas de données à afficher.")
    else:
        pool = filtered if len(filtered) > 0 else results_df
        ticker_list = pool["Ticker"].tolist()
        name_by_ticker = dict(zip(pool["Ticker"], pool["Nom"]))
        selected_ticker = st.selectbox(
            "Sélectionner un titre :", ticker_list,
            format_func=lambda t: f"{name_by_ticker.get(t, t)} ({t})",
        )
        stock_row = results_df[results_df["Ticker"] == selected_ticker].iloc[0]

        range_labels = {"1M": "1 mois", "6M": "6 mois", "YTD": "YTD",
                         "1A": "1 an", "5A": "5 ans", "Tout": "Tout"}
        selected_range = st.radio(
            "Plage", list(range_labels.keys()), format_func=lambda k: range_labels[k],
            horizontal=True, key="chart_range", index=3,
        )

        with st.spinner("Chargement de l'historique..."):
            full_history = fetch_chart_history(selected_ticker)
        stock_data = slice_by_range(full_history, selected_range) if not full_history.empty else stock_row["_history"]
        if stock_data.empty:
            stock_data = stock_row["_history"]
        ind = compute_indicators(stock_data)

        short_history = len(stock_data) < 20
        if short_history:
            st.info(
                f"Historique disponible pour {selected_ticker} : {len(stock_data)} séance(s) seulement "
                f"(cotation récente ou plage \"{range_labels[selected_range]}\" plus longue que l'historique réel). "
                "Bandes de Bollinger, SMA 20 et RSI nécessitent au moins 20 séances et peuvent ne pas s'afficher."
            )

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.55, 0.2, 0.25],
        )
        fig.add_trace(go.Candlestick(
            x=stock_data.index, open=stock_data["Open"], high=stock_data["High"],
            low=stock_data["Low"], close=stock_data["Close"], name="Prix",
            increasing_line_color="#22C55E", decreasing_line_color="#F0466E",
        ), row=1, col=1)
        if ind["boll_up"].notna().any():
            fig.add_trace(go.Scatter(x=stock_data.index, y=ind["boll_up"], name="Bollinger haut",
                                      line=dict(color="rgba(108,142,255,.5)", width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=ind["boll_low"], name="Bollinger bas",
                                      line=dict(color="rgba(108,142,255,.5)", width=1),
                                      fill="tonexty", fillcolor="rgba(108,142,255,.06)"), row=1, col=1)
        if ind["sma20"].notna().any():
            fig.add_trace(go.Scatter(x=stock_data.index, y=ind["sma20"], name="SMA 20",
                                      line=dict(color="#F5A623", width=1.3)), row=1, col=1)

        vol_colors = np.where(stock_data["Close"] >= stock_data["Open"], "#22C55E", "#F0466E")
        fig.add_trace(go.Bar(x=stock_data.index, y=stock_data["Volume"], name="Volume",
                              marker_color=vol_colors, showlegend=False), row=2, col=1)

        if ind["rsi"].notna().any():
            fig.add_trace(go.Scatter(x=stock_data.index, y=ind["rsi"], name="RSI (14)",
                                      line=dict(color="#6C8EFF", width=1.5)), row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#22C55E", row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#F0466E", row=3, col=1)

        fig.update_layout(
            height=720, xaxis_rangeslider_visible=False, showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E7ECF5"}, margin=dict(l=10, r=10, t=44, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            hovermode="x unified",
        )
        # Masque les week-ends (et jours fériés, absents des données) sur l'axe des temps :
        # sans cela, Plotly traite l'axe X comme du temps continu et laisse un "trou" visuel
        # chaque lundi/après-jour-férié, ce qui casse la lisibilité des chandeliers.
        if len(stock_data) > 1:
            all_days = pd.date_range(start=stock_data.index.min(), end=stock_data.index.max(), freq="D")
            present = set(stock_data.index.normalize())
            missing_weekdays = [d for d in all_days if d.weekday() < 5 and d not in present]
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=missing_weekdays)])
        fig.update_xaxes(gridcolor="#232C42")
        fig.update_yaxes(gridcolor="#232C42")
        fig.update_yaxes(title_text=f"Prix ({stock_row['Devise']})", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI (14)", range=[0, 100], row=3, col=1)

        with st.container(border=True):
            st.markdown(
                f'<div class="panel-title">{name_by_ticker.get(selected_ticker, selected_ticker)} : '
                f'prix, volume et RSI</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="panel-sub">Plage : {range_labels[selected_range]} · Bandes de Bollinger (20j, 2σ) '
                'et SMA 20 en superposition du prix, seuils de survente/surachat à 30 et 70 sur le RSI. '
                'Week-ends et jours fériés masqués sur l\'axe des temps.</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, width="stretch", key="main_chart")

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        range_pct = None
        if len(stock_data) >= 2:
            first_close = float(stock_data["Close"].iloc[0])
            last_close = float(stock_data["Close"].iloc[-1])
            if first_close:
                range_pct = (last_close - first_close) / first_close * 100
        if range_pct is None:
            evo_value, evo_color = "N/A", "var(--text-hi)"
        else:
            evo_value = f"{range_pct:+.2f} %"
            evo_color = "var(--rose)" if range_pct < 0 else "var(--emerald)"

        pe_chart = fetch_pe_ratio(selected_ticker)
        chart_kpis = [
            ("Prix", f"{stock_row['Devise']}{stock_row['Prix']:.2f}", "var(--text-hi)"),
            (f"Évolution ({range_labels[selected_range]})", evo_value, evo_color),
            ("RSI (14)", str(stock_row["RSI (14)"]) if stock_row["RSI (14)"] is not None else "N/A", "var(--text-hi)"),
            ("Ratio Volume", f"{stock_row['Ratio Vol.']}x" if stock_row["Ratio Vol."] else "N/A", "var(--text-hi)"),
            ("Score Opportunité", f"{stock_row['Score Opp.']} / 100", "var(--text-hi)"),
            ("P/E", f"{pe_chart}x" if pe_chart is not None else "N/A", "var(--text-hi)"),
        ]
        chart_kpi_html = "".join(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{color}">{value}</div></div>'
            for label, value, color in chart_kpis
        )
        st.markdown(f'<div class="kpi-grid">{chart_kpi_html}</div>', unsafe_allow_html=True)

# ---- Onglet Méthodologie -------------------------------------------------
with tab_about:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Comment le score d\'opportunité est calculé</div>', unsafe_allow_html=True)
        st.markdown(
            """
            | Signal | Logique | Points max |
            |---|---|---|
            | RSI (14) | rampe continue : 25 pts à RSI ≤ 15, 0 pt à RSI ≥ 50 | 25 |
            | Bandes de Bollinger | proportionnel à la profondeur sous la bande basse (20j, 2σ) | 20 |
            | Momentum récent | variation 1J et 5J : plus la baisse récente est marquée, plus le score est élevé ; un rebond récent n'ajoute rien | 20 |
            | Volume | ratio vs moyenne 20j, réduit de 70% si la séance est en forte hausse (un volume élevé en hausse n'est pas un signal d'opportunité) | 15 |
            | MACD | croisement haussier naissant de l'histogramme | 10 |
            | Range période | rampe continue selon la proximité du plus bas sur 1 an | 10 |
            """
        )
        st.caption(
            "Le score est plafonné à 100. Les composantes RSI, Bollinger et Range évoluent en continu, "
            "et le momentum récent est explicitement pris en compte : un titre "
            "qui rebondit fortement voit son score baisser même si RSI/Bollinger n'ont pas encore rattrapé "
            "ce rebond. Le score combine des signaux de survente et de retournement : il ne constitue en "
            "aucun cas une garantie de rebond futur."
        )
        st.caption(
            "Le P/E (cours/bénéfices), affiché sur les cartes et l'onglet graphique, est volontairement "
            "indicatif et non intégré au score : c'est un signal de valorisation fondamentale, pas de survente "
            "technique, il n'est comparable qu'au sein d'un même secteur, et il est indéfini pour les sociétés "
            "sans bénéfices (biotechs, valeurs de croissance en phase d'investissement), ce qui le rendrait peu "
            "fiable comme critère automatique."
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="panel-title">Marchés disponibles</div>', unsafe_allow_html=True)
        for m in MARKETS.values():
            tag = ", liste maison, non-officielle" if m.is_curated else ", composition officielle (Wikipedia, temps réel)"
            st.markdown(f"- **{m.label}**{tag}")

st.markdown(
    """
    <div class="disclaimer">
    Les résultats fournis par cette application reposent sur des indicateurs techniques
    (RSI, Bandes de Bollinger, MACD, ratio de volume) et sont présentés à titre purement informatif
    et pédagogique. Ils ne constituent en aucun cas un conseil en investissement, une recommandation
    d'achat ou de vente, ni une garantie de résultat. Les marchés actions et cryptomonnaies comportent
    des risques significatifs, y compris la perte totale du capital investi. Les calculs peuvent
    contenir des approximations ou retards inhérents aux données (Yahoo Finance, Wikipedia).
    Consultez un professionnel agréé avant toute décision d'investissement.
    </div>
    """,
    unsafe_allow_html=True,
)
