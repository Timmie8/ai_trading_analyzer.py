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
    df['Target_1W'] = (df['Close'].shift(-5) > df['Close']).astype(int)   # Korte termijn: 1 week
    df['Target_1Y'] = (df['Close'].shift(-252) > df['Close']).astype(int)  # Lange termijn: 1 jaar
    df['Target_2Y'] = (df['Close'].shift(-504) > df['Close']).astype(int)  # Lange termijn: 2 jaar

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

# Styling functie voor tabelkleuren (Groen = Bullish, Rood = Bearish)
def highlight_signals(val):
    if val == "BULLISH":
        return 'background-color: #d4edda; color: #155724; font-weight: bold;'
    elif val == "BEARISH":
        return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
    return ''

def highlight_prob(val):
    try:
        prob = float(val.replace('%', ''))
        if prob >= 55.0:
            return 'color: #28a745; font-weight: bold;'
        elif prob <= 45.0:
            return 'color: #dc3545; font-weight: bold;'
    except:
        pass
    return ''

# Sidebar voor instellingen
ticker_input = st.sidebar.text_input("Voer Ticker in:", value="NVDA").upper()
start_button = st.sidebar.button("Start AI Analyse")

if start_button:
    with st.spinner(f"Bezig met ophalen van data en trainen van AI-modellen voor {ticker_input}..."):
        try:
            df = get_stock_data(ticker_input)
            
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'MACD', 'RSI', 'Return_1D']
            X = df[features]
            
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            results = []
            korte_termijn_probs = []
            lange_termijn_probs = []

            # =========================================================
            # 1. KORTE TERMIJN MODELLEN (6 AI MODELLEN - 1 WEEK)
            # =========================================================
            y_1w = df['Target_1W']
            
            # Model 1: LSTM Deep Learning
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
                    "Categorie": "Korte Termijn",
                    "Model Name": "LSTM Deep Learning",
                    "Horizon": "1 Week",
                    "Signaal": "BULLISH" if prob_lstm > 0.5 else "BEARISH",
                    "Stijgingskans (%)": f"{prob_lstm * 100:.1f}%"
                })
                korte_termijn_probs.append(prob_lstm)

            # Model 2: XGBoost
            if len(np.unique(y_1w.iloc[:-5])) > 1:
                xgb_mod = xgb.XGBClassifier(eval_metric='logloss', max_depth=3, n_estimators=50)
                xgb_mod.fit(X_scaled[:-5], y_1w.iloc[:-5])
                prob_xgb = safe_predict_proba(xgb_mod, X_scaled[-1:])
                results.append({
                    "Categorie": "Korte Termijn",
                    "Model Name": "XGBoost Classifier",
                    "Horizon": "1 Week",
                    "Signaal": "BULLISH" if prob_xgb > 0.5 else "BEARISH",
                    "Stijgingskans (%)": f"{prob_xgb * 100:.1f}%"
                })
                korte_termijn_probs.append(prob_xgb)

            # Model 3: LightGBM
            if len(np.unique(y_1w.iloc[:-5])) > 1:
                lgbm = lgb.LGBMClassifier(verbosity=-1, n_estimators=50)
                lgbm.fit(X_scaled[:-5], y_1w.iloc[:-5])
                prob_lgb = safe_predict_proba(lgbm, X_scaled[-1:])
                results.append({
                    "Categorie": "Korte Termijn",
                    "Model Name": "LightGBM",
                    "Horizon": "1 Week",
                    "Signaal": "BULLISH" if prob_lgb > 0.5 else "BEARISH",
                    "Stijgingskans (%)": f"{prob_lgb * 100:.1f}%"
                })
                korte_termijn_probs.append(prob_lgb)

            # Model 4: MLP Neural Network
            if len(np.unique(y_1w.iloc[:-5])) > 1:
                mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, random_state=42)
                mlp.fit(X_scaled[:-5], y_1w.iloc[:-5])
                prob_mlp = safe_predict_proba(mlp, X_scaled[-1:])
                results.append({
                    "Categorie": "Korte Termijn",
                    "Model Name": "MLP Neural Network",
                    "Horizon": "1 Week",
                    "Signaal": "BULLISH" if prob_mlp > 0.5 else "BEARISH",
                    "Stijgingskans (%)": f"{prob_mlp * 100:.1f}%"
                })
                korte_termijn_probs.append(prob_mlp)

            # Model 5: Logistic Regression
            if len(np.unique(y_1w.iloc[:-5])) > 1:
                lr = LogisticRegression()
                lr.fit(X_scaled[:-5], y_1w.iloc[:-5])
                prob_lr = safe_predict_proba(lr, X_scaled[-1:])
                results.append({
                    "Categorie": "Korte Termijn",
                    "Model Name": "Logistic Regression",
                    "Horizon": "1 Week",
                    "Signaal": "BULLISH" if prob_lr > 0.5 else "BEARISH",
                    "Stijgingskans (%)": f"{prob_lr * 100:.1f}%"
                })
                korte_termijn_probs.append(prob_lr)

            # Model 6: Support Vector Machine (SVM)
            if len(np.unique(y_1w.iloc[:-5])) > 1:
                svm = SVC(probability=True, kernel='rbf')
                svm.fit(X_scaled[:-5], y_1w.iloc[:-5])
                prob_svm = safe_predict_proba(svm, X_scaled[-1:])
                results.append({
                    "Categorie": "Korte Termijn",
                    "Model Name": "Support Vector Machine (SVM)",
                    "Horizon": "1 Week",
                    "Signaal": "BULLISH" if prob_svm > 0.5 else "BEARISH",
                    "Stijgingskans (%)": f"{prob_svm * 100:.1f}%"
                })
                korte_termijn_probs.append(prob_svm)

            # =========================================================
            # 2. LANGE TERMIJN MODELLEN (RANDOM FOREST & GRADIENT BOOSTING)
            # =========================================================
            # Model 7: Random Forest Ensemble (1 Jaar Horizon)
            y_1y = df['Target_1Y']
            y_1y_train = y_1y.iloc[:-252]
            if len(y_1y_train) > 100 and len(np.unique(y_1y_train)) > 1:
                rf_1y = RandomForestClassifier(n_estimators=50, random_state=42)
                rf_1y.fit(X_scaled[:-252], y_1y_train)
                prob_rf_1y = safe_predict_proba(rf_1y, X_scaled[-1:])
                results.append({
                    "Categorie": "Lange Termijn",
                    "Model Name": "Random Forest Ensemble",
                    "Horizon": "1 Jaar",
                    "Signaal": "BULLISH" if prob_rf_1y > 0.5 else "BEARISH",
                    "Stijgingskans (%)": f"{prob_rf_1y * 100:.1f}%"
                })
                lange_termijn_probs.append(prob_rf_1y)

            # Model 8: Gradient Boosting Ensemble (2 Jaar Horizon)
            y_2y = df['Target_2Y']
            y_2y_train = y_2y.iloc[:-504]
            if len(y_2y_train) > 100 and len(np.unique(y_2y_train)) > 1:
                gb_2y = GradientBoostingClassifier(n_estimators=50, random_state=42)
                gb_2y.fit(X_scaled[:-504], y_2y_train)
                prob_gb_2y = safe_predict_proba(gb_2y, X_scaled[-1:])
                results.append({
                    "Categorie": "Lange Termijn",
                    "Model Name": "Gradient Boosting Ensemble",
                    "Horizon": "2 Jaar",
                    "Signaal": "BULLISH" if prob_gb_2y > 0.5 else "BEARISH",
                    "Stijgingskans (%)": f"{prob_gb_2y * 100:.1f}%"
                })
                lange_termijn_probs.append(prob_gb_2y)

            # =========================================================
            # WEERGEVEN SCORE & RESULTATEN OP SCHERM
            # =========================================================
            st.subheader(f"Analyse Resultaten voor {ticker_input}")
            
            col1, col2 = st.columns(2)

            # 1. Korte Termijn Score Box
            if korte_termijn_probs:
                kt_avg = np.mean(korte_termijn_probs) * 100
                col1.metric("⚡ KORTE TERMIJN SCORE (1 WLEK)", f"{kt_avg:.1f}%")
                if kt_avg > 60:
                    col1.success(f"**Korte Termijn Signaal:** STERK BULLISH ({kt_avg:.1f}%)")
                elif kt_avg >= 45:
                    col1.warning(f"**Korte Termijn Signaal:** NEUTRAAL ({kt_avg:.1f}%)")
                else:
                    col1.error(f"**Korte Termijn Signaal:** BEARISH ({kt_avg:.1f}%)")

            # 2. Lange Termijn Score Box
            if lange_termijn_probs:
                lt_avg = np.mean(lange_termijn_probs) * 100
                col2.metric("🏛️ LANGE TERMIJN SCORE (1-2 JAAR)", f"{lt_avg:.1f}%")
                if lt_avg > 60:
                    col2.success(f"**Lange Termijn Signaal:** STERK BULLISH ({lt_avg:.1f}%)")
                elif lt_avg >= 45:
                    col2.warning(f"**Lange Termijn Signaal:** NEUTRAAL ({lt_avg:.1f}%)")
                else:
                    col2.error(f"**Lange Termijn Signaal:** BEARISH ({lt_avg:.1f}%)")

            st.write("---")
            st.write("### AI Model Overzicht per Categorie")
            
            results_df = pd.DataFrame(results)
            
            # Styling voor Groen (Bullish) en Rood (Bearish)
            styled_df = results_df.style.map(highlight_signals, subset=['Signaal'])\
                                        .map(highlight_prob, subset=['Stijgingskans (%)'])
            
            st.dataframe(styled_df, use_container_width=True)

        except Exception as e:
            st.error(f"Er is een fout opgetreden bij het verwerken van {ticker_input}: {e}")
else:
    st.info("Voer een ticker in de linkerbalk in en klik op 'Start AI Analyse' om het proces te starten.")
