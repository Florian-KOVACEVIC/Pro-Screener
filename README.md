# Screener-bourse

# Market Screener & Short-Term Opportunities Dashboard

Live demo and interactive financial screener built to detect daily short-term trading opportunities, oversold stocks, and bounce signals across S&P 500 sectors.

Built as a personal project to explore quantitative finance, technical analysis, and automated market screening through hands-on development.

---

## Features

### Daily Market Screener
* **Automated Scraping**: Dynamic daily fetching of the S&P 500 stock universe from Wikipedia and Yahoo Finance (`yfinance`).
* **Sector Filtering**: Filter opportunities by GICS sectors (Technology, Healthcare, Financials, etc.).
* **Custom Quantitative Score (0–100)**: Proprietary opportunity rating combining RSI level, Bollinger Band lower-bound breach, and abnormal volume spikes.
* **Volume Anomaly Detection**: Highlights stocks experiencing unusual trading activity compared to their 20-day average volume.

### Technical Analysis
* **Core Indicators**: Relative Strength Index (RSI 14), Bollinger Bands (20 days, 2 std dev), Volume Ratio, and Moving Averages.
* **Multi-Period Performance**: Real-time tracking of 1-day and 5-day performance changes.
* **Interactive Candlestick Charts**: Built-in Plotly charts displaying price action and indicator subplots without leaving the app.

### Practical Utilities & UX
* **Export Capabilities**: One-click CSV export of filtered opportunities for further analysis.
* **Daily Caching**: Integrated Streamlit data caching (`@st.cache_data`) for fast load times and optimized API calls.
* **Interactive Data Tables**: Responsive data grids with progress bars for scoring and formatted metric cards.

---

## Tech Stack

| Layer | Tools |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Framework** | Streamlit |
| **Market Data** | `yfinance` (Yahoo Finance API) |
| **Visualization** | Plotly (Interactive Candlestick & Subplot Charts) |
| **Data & Computation** | Pandas, NumPy |

---

## Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/Florian-KOVACEVIC/screener-bourse.git](https://github.com/Florian-KOVACEVIC/screener-bourse.git)
cd screener-bourse

# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

python -m streamlit run app4.py
