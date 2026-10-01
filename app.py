import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

# Pagina instellingen
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")
st.title("📈 AI & ML Trading Analyzer")

def get_stock_data(ticker_symbol, period="5y"):
    df = yf.download(ticker_symbol, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Technische Indicatoren
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Returns & Targets
    df['Return_1D'] = df['Close'].pct_change(1)
    df['Target_1W'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    df['Target_1Y'] = (df['Close'].shift(-252) > df['Close']).astype(int)
    df['Target_2Y'] = (df['Close'].shift(-504) > df['Close']).astype(int)

    df.dropna(inplace=True)
    return df

def safe_predict_proba(model, X_input):
    probs = model.predict_proba(X_input)[0]
    classes = list(model.classes_)
    if 1 in classes:
        idx_1 = classes.index(1)
        return float(probs[idx_1])
    else:
        return 0.0

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=32, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Styling functie voor de GEHELE regel
HOOFDVOORSPELLERS = [
    "XGBoost Classifier", 
    "LightGBM", 
    "LSTM Deep Learning"
]

def style_row(row):
    model_name = str(row['Model Name']).strip()
    signaal = str(row['Signaal']).strip().upper()
    
    if signaal == "BULLISH":
        if model_name in HOOFDVOORSPELLERS:
            return ['background-color: #1e7e34; color: white; font-weight: bold;'] * len(row)
        else:
            return ['background-color: #d4edda; color: #155724; font-weight: bold;'] * len(row)
            
    elif signaal == "BEARISH":
        return ['background-color: #f8d7da; color: #721c24; font-weight: bold;'] * len(row)
        
    return [''] * len(row)

def run_single_stock_analysis(ticker_symbol):
    """
    Voert exact dezelfde AI-analyse uit voor een enkele ticker en retourneert 
    de gedetailleerde resultaten en het korte termijn gemiddelde.
    """
    df = get_stock_data(ticker_symbol)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'MACD', 'RSI', 'Return_1D']
    X = df[features]
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    korte_termijn_probs = []

    y_1w = df['Target_1W']
    
    # 1. LSTM Deep Learning
    X_lstm, y_lstm = [], []
    time_step = 10
    for i in range(time_step, len(X_scaled) - 5):
        X_lstm.append(X_scaled[i-time_step:i])
        y_lstm.append(y_1w.iloc[i])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    
    if len(X_lstm) > 50 and len(np.unique(y_lstm)) > 1:
        lstm = build_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
        lstm.fit(X_lstm[:-10], y_lstm[:-10], epochs=10, batch_size=32, verbose=0)
        last_seq = np.expand_dims(X_scaled[-time_step:], axis=0)
        prob_lstm = float(lstm.predict(last_seq, verbose=0)[0][0])
        results.append({
            "Model Name": "LSTM Deep Learning",
            "Focus / Horizon": "1 Week (Korte Termijn)",
            "Signaal": "BULLISH" if prob_lstm > 0.5 else "BEARISH",
            "Stijgingskans (%)": f"{prob_lstm * 100:.1f}%"
        })
        korte_termijn_probs.append(prob_lstm)

    # 2. XGBoost
    if len(np.unique(y_1w.iloc[:-5])) > 1:
        xgb_mod = xgb.XGBClassifier(eval_metric='logloss', max_depth=3, n_estimators=50)
        xgb_mod.fit(X_scaled[:-5], y_1w.iloc[:-5])
        prob_xgb = safe_predict_proba(xgb_mod, X_scaled[-1:])
        results.append({
            "Model Name": "XGBoost Classifier",
            "Focus / Horizon": "1 Week (Swingtrade)",
            "Signaal": "BULLISH" if prob_xgb > 0.5 else "BEARISH",
            "Stijgingskans (%)": f"{prob_xgb * 100:.1f}%"
        })
        korte_termijn_probs.append(prob_xgb)

    # 3. LightGBM
    if len(np.unique(y_1w.iloc[:-5])) > 1:
        lgbm = lgb.LGBMClassifier(verbosity=-1, n_estimators=50)
        lgbm.fit(X_scaled[:-5], y_1w.iloc[:-5])
        prob_lgb = safe_predict_proba(lgbm, X_scaled[-1:])
        results.append({
            "Model Name": "LightGBM",
            "Focus / Horizon": "1 Week (Korte Termijn)",
            "Signaal": "BULLISH" if prob_lgb > 0.5 else "BEARISH",
            "Stijgingskans (%)": f"{prob_lgb * 100:.1f}%"
        })
        korte_termijn_probs.append(prob_lgb)

    # 4. MLP Neural Net
    if len(np.unique(y_1w.iloc[:-5])) > 1:
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, random_state=42)
        mlp.fit(X_scaled[:-5], y_1w.iloc[:-5])
        prob_mlp = safe_predict_proba(mlp, X_scaled[-1:])
        results.append({
            "Model Name": "MLP Neural Network",
            "Focus / Horizon": "1 Week (Korte Termijn)",
            "Signaal": "BULLISH" if prob_mlp > 0.5 else "BEARISH",
            "Stijgingskans (%)": f"{prob_mlp * 100:.1f}%"
        })
        korte_termijn_probs.append(prob_mlp)

    # 5. Logistic Regression
    if len(np.unique(y_1w.iloc[:-5])) > 1:
        lr = LogisticRegression()
        lr.fit(X_scaled[:-5], y_1w.iloc[:-5])
        prob_lr = safe_predict_proba(lr, X_scaled[-1:])
        results.append({
            "Model Name": "Logistic Regression",
            "Focus / Horizon": "1 Week (Sentiment)",
            "Signaal": "BULLISH" if prob_lr > 0.5 else "BEARISH",
            "Stijgingskans (%)": f"{prob_lr * 100:.1f}%"
        })
        korte_termijn_probs.append(prob_lr)

    # 6. Support Vector Machine (SVM)
    if len(np.unique(y_1w.iloc[:-5])) > 1:
        svm = SVC(probability=True, kernel='rbf')
        svm.fit(X_scaled[:-5], y_1w.iloc[:-5])
        prob_svm = safe_predict_proba(svm, X_scaled[-1:])
        results.append({
            "Model Name": "Support Vector Machine (SVM)",
            "Focus / Horizon": "1-2 Weken (Trend)",
            "Signaal": "BULLISH" if prob_svm > 0.5 else "BEARISH",
            "Stijgingskans (%)": f"{prob_svm * 100:.1f}%"
        })
        korte_termijn_probs.append(prob_svm)

    # 7. Random Forest (1 Jaar Horizon)
    y_1y = df['Target_1Y']
    y_1y_train = y_1y.iloc[:-252]
    if len(y_1y_train) > 100 and len(np.unique(y_1y_train)) > 1:
        rf_1y = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_1y.fit(X_scaled[:-252], y_1y_train)
        prob_rf_1y = safe_predict_proba(rf_1y, X_scaled[-1:])
        results.append({
            "Model Name": "Random Forest Ensemble",
            "Focus / Horizon": "1 Jaar (Middellang)",
            "Signaal": "BULLISH" if prob_rf_1y > 0.5 else "BEARISH",
            "Stijgingskans (%)": f"{prob_rf_1y * 100:.1f}%"
        })

    # 8. Gradient Boosting (2 Jaar Horizon)
    y_2y = df['Target_2Y']
    y_2y_train = y_2y.iloc[:-504]
    if len(y_2y_train) > 100 and len(np.unique(y_2y_train)) > 1:
        gb_2y = GradientBoostingClassifier(n_estimators=50, random_state=42)
        gb_2y.fit(X_scaled[:-504], y_2y_train)
        prob_gb_2y = safe_predict_proba(gb_2y, X_scaled[-1:])
        results.append({
            "Model Name": "Gradient Boosting Ensemble",
            "Focus / Horizon": "2 Jaar (Langetermijn)",
            "Signaal": "BULLISH" if prob_gb_2y > 0.5 else "BEARISH",
            "Stijgingskans (%)": f"{prob_gb_2y * 100:.1f}%"
        })

    korte_termijn_avg = np.mean(korte_termijn_probs) * 100 if korte_termijn_probs else 0.0
    return pd.DataFrame(results), korte_termijn_avg

# Sidebar voor enkelvoudige instellingen
st.sidebar.header("Enkelvoudige Analyse")
ticker_input = st.sidebar.text_input("Voer Ticker in:", value="NVDA").upper()
start_button = st.sidebar.button("Start AI Analyse")

# Enkelvoudige Analyse Uitvoering
if start_button:
    with st.spinner(f"Bezig met ophalen van data en trainen van AI-modellen voor {ticker_input}..."):
        try:
            results_df, korte_termijn_avg = run_single_stock_analysis(ticker_input)

            st.subheader(f"Analyse Resultaten voor {ticker_input}")
            
            if not results_df.empty:
                col1, col2 = st.columns(2)
                col1.metric("Totale AI Korte Termijn Kans (1 Week)", f"{korte_termijn_avg:.1f}%")
                
                if korte_termijn_avg > 60:
                    st.success(f"**AI Signaal:** STERK BULLISH (KOPEN) — Korte termijn consensus is {korte_termijn_avg:.1f}%")
                elif korte_termijn_avg >= 45:
                    st.warning(f"**AI Signaal:** NEUTRAAL / WATCH — Korte termijn consensus is {korte_termijn_avg:.1f}%")
                else:
                    st.error(f"**AI Signaal:** BEARISH (VERKOPEN) — Korte termijn consensus is {korte_termijn_avg:.1f}%")

            st.write("---")
            st.write("### AI Model Overzicht")
            
            styled_df = results_df.style.apply(style_row, axis=1)
            st.dataframe(styled_df, use_container_width=True)

        except Exception as e:
            st.error(f"Er is een fout opgetreden bij het verwerken van {ticker_input}: {e}")

# ==============================================================================
# BATCH ANALYSE SECTIE (510+ AANDELEN TEGELIJK)
# ==============================================================================
st.write("---")
st.header("📊 Multi-Aandelen Batch Analyzer (tot 510 aandelen)")
st.caption("Voer een lijst van tickers in (gescheiden door komma's, spaties of nieuwe regels) om een volledige AI-analyse uit te voeren op meerdere aandelen tegelijk.")

default_batch_tickers = "NVDA, AAPL, MSFT, AMZN, GOOGL, TSLA, META"
batch_input = st.text_area("Voer tickers in (max. 510):", value=default_batch_tickers, height=120)

start_batch_button = st.button("Start Batch AI Analyse")

if start_batch_button:
    # Tickers verwerken uit de input (verwijder komma's, spaties en lege regels)
    raw_list = batch_input.replace(',', ' ').split()
    tickers_list = list(dict.fromkeys([t.strip().upper() for t in raw_list if t.strip()]))
    
    if len(tickers_list) > 510:
        st.warning(f"Er zijn {len(tickers_list)} tickers ingevoerd. De lijst is automatisch ingekort tot de eerste 510 aandelen.")
        tickers_list = tickers_list[:510]
        
    if not tickers_list:
        st.error("Geen geldige tickers ingevoerd.")
    else:
        st.info(f"Starten van AI-analyse voor {len(tickers_list)} aandelen...")
        
        batch_summary = []
        detailed_results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_tickers = len(tickers_list)
        
        for idx, ticker in enumerate(tickers_list):
            status_text.text(f"Bezig met verwerken van {ticker} ({idx + 1}/{total_tickers})...")
            try:
                res_df, avg_prob = run_single_stock_analysis(ticker)
                
                # Bepaal consensus signaal
                if avg_prob > 60:
                    consensus = "STERK BULLISH"
                elif avg_prob >= 45:
                    consensus = "NEUTRAAL"
                else:
                    consensus = "BEARISH"
                
                batch_summary.append({
                    "Ticker": ticker,
                    "Korte Termijn Kans (1 Wk)": f"{avg_prob:.1f}%",
                    "Consensus Signaal": consensus,
                    "Aantal Modellen Evaluatie": len(res_df)
                })
                detailed_results[ticker] = res_df
            except Exception as e:
                batch_summary.append({
                    "Ticker": ticker,
                    "Korte Termijn Kans (1 Wk)": "N/A",
                    "Consensus Signaal": "FOUT / GEEN DATA",
                    "Aantal Modellen Evaluatie": 0
                })
            
            progress_bar.progress((idx + 1) / total_tickers)
            
        status_text.text("Batch analyse afgerond!")
        
        # Samenvattingstabel
        summary_df = pd.DataFrame(batch_summary)
        
        def highlight_summary(val):
            if val == "STERK BULLISH":
                return 'background-color: #1e7e34; color: white; font-weight: bold;'
            elif val == "NEUTRAAL":
                return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
            elif val == "BEARISH":
                return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
            return ''

        st.subheader("📋 Batch Overzicht & Consensus")
        styled_summary = summary_df.style.map(highlight_summary, subset=['Consensus Signaal'])
        st.dataframe(styled_summary, use_container_width=True)
        
       # Details per aandeel
        st.subheader("🔍 Gedetailleerde AI Resultaten per Aandeel")
        for ticker in tickers_list:
            if ticker in detailed_results and not detailed_results[ticker].empty:
                with st.expander(f"Bekijk gedetailleerde AI-modellen voor: {ticker}"):
                    styled_detail = detailed_results[ticker].style.apply(style_row, axis=1)
                    st.dataframe(styled_detail, use_container_width=True)
            else:
                with st.expander(f"Bekijk gedetailleerde AI-modellen voor: {ticker}"):
                    st.write("Geen gedetailleerde gegevens beschikbaar voor deze ticker.")

### Wat is er toegevoegd?
1. **Volledige functionaliteit behouden**: De individuele analyse bovenaan gebruikt exact dezelfde AI-modellen (LSTM, XGBoost, LightGBM, MLP, Logistic Regression, SVM, Random Forest, Gradient Boosting) en styling.
2. **Batch-sectie onderaan**:
   * Een tekstvak waarin je maximaal 510 tickers kunt plakken (bijv. gekopieerd uit Excel of een tekstbestand).
   * Voortgangsbalk (`progress_bar`) zodat je de status van de berekeningen per aandeel kunt volgen.
   * **Rangschikking & Samenvatting**: Een overzichtstabel met alle geanalyseerde aandelen, hun 1-weekse consensus-kans en signaal.
   * **Inklapbare detailweergave**: Je kunt per aandeel doorklikken om de exacte tabel met de donkergroene/lichtgroene/rode accentueringsregels in te zien.
