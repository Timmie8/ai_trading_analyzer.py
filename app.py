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
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

def get_stock_data(ticker_symbol, period="3y"):
    """Haalt data op via yfinance en voegt technische indicatoren toe."""
    df = yf.download(ticker_symbol, period=period, interval="1d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Technische Indicatoren berekenen
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
    df['Target_1W'] = (df['Close'].shift(-5) > df['Close']).astype(int)  # 1 week (korte termijn)
    df['Target_1Y'] = (df['Close'].shift(-252) > df['Close']).astype(int) # 1 jaar
    df['Target_2Y'] = (df['Close'].shift(-504) > df['Close']).astype(int) # 2 jaar

    df.dropna(inplace=True)
    return df

def build_lstm_model(input_shape):
    """LSTM Neuraal Netwerk voor hele korte termijn (1 week)."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def analyze_stock_ai(ticker_symbol):
    print(f"--- AI ANALYSE AANDIEL: {ticker_symbol} ---")
    df = get_stock_data(ticker_symbol)
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'MACD', 'RSI', 'Return_1D']
    X = df[features]
    
    # Normalisatie voor AI modellen
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []

    # -------------------------------------------------------------
    # 1. KORTE TERMIJN MODELLEN (1-5 DAGEN / SWINGTRADE FOCUS)
    # -------------------------------------------------------------
    y_1w = df['Target_1W']
    
    # Model 1: LSTM Deep Learning (Korte Termijn / 1 Week)
    X_lstm = []
    y_lstm = []
    time_step = 10
    for i in range(time_step, len(X_scaled) - 5):
        X_lstm.append(X_scaled[i-time_step:i])
        y_lstm.append(y_1w.iloc[i])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    
    if len(X_lstm) > 50:
        lstm = build_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
        lstm.fit(X_lstm[:-10], y_lstm[:-10], epochs=15, batch_size=16, verbose=0)
        last_seq = np.expand_dims(X_scaled[-time_step:], axis=0)
        prob_lstm = float(lstm.predict(last_seq, verbose=0)[0][0])
        results.append({
            "Model Name": "LSTM Deep Learning",
            "Focus / Horizon": "1 Week (Korte Termijn)",
            "Bullish / Bearish": "BULLISH" if prob_lstm > 0.5 else "BEARISH",
            "Stijgingskans (%)": f"{prob_lstm * 100:.1f}%"
        })
    
    # Model 2: XGBoost Classifier (Korte Termijn Momentum)
    xgb_mod = xgb.XGBClassifier(eval_metric='logloss', max_depth=3, n_estimators=100)
    xgb_mod.fit(X_scaled[:-5], y_1w.iloc[:-5])
    prob_xgb = xgb_mod.predict_proba(X_scaled[-1:]) [0][1]
    results.append({
        "Model Name": "XGBoost Classifier",
        "Focus / Horizon": "1 Week (Swingtrade)",
        "Bullish / Bearish": "BULLISH" if prob_xgb > 0.5 else "BEARISH",
        "Stijgingskans (%)": f"{prob_xgb * 100:.1f}%"
    })

    # Model 3: LightGBM (Korte Termijn Volatiliteit)
    lgbm = lgb.LGBMClassifier(verbosity=-1, n_estimators=100)
    lgbm.fit(X_scaled[:-5], y_1w.iloc[:-5])
    prob_lgb = lgbm.predict_proba(X_scaled[-1:]) [0][1]
    results.append({
        "Model Name": "LightGBM",
        "Focus / Horizon": "1 Week (Korte Termijn)",
        "Bullish / Bearish": "BULLISH" if prob_lgb > 0.5 else "BEARISH",
        "Stijgingskans (%)": f"{prob_lgb * 100:.1f}%"
    })

    # Model 4: Multi-Layer Perceptron (MLP Neural Net)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
    mlp.fit(X_scaled[:-5], y_1w.iloc[:-5])
    prob_mlp = mlp.predict_proba(X_scaled[-1:]) [0][1]
    results.append({
        "Model Name": "MLP Neural Network",
        "Focus / Horizon": "1 Week (Korte Termijn)",
        "Bullish / Bearish": "BULLISH" if prob_mlp > 0.5 else "BEARISH",
        "Stijgingskans (%)": f"{prob_mlp * 100:.1f}%"
    })

    # Model 5: Logistic Regression (Statistisch Sentiment)
    lr = LogisticRegression()
    lr.fit(X_scaled[:-5], y_1w.iloc[:-5])
    prob_lr = lr.predict_proba(X_scaled[-1:]) [0][1]
    results.append({
        "Model Name": "Logistic Regression",
        "Focus / Horizon": "1 Week (Sentiment)",
        "Bullish / Bearish": "BULLISH" if prob_lr > 0.5 else "BEARISH",
        "Stijgingskans (%)": f"{prob_lr * 100:.1f}%"
    })

    # -------------------------------------------------------------
    # 2. ENSEMBLE MODELLEN (1 JAAR & 2 JAAR HORIZON)
    # -------------------------------------------------------------
    # Model 6: Random Forest Ensemble (1 Jaar Horizon)
    y_1y = df['Target_1Y']
    if len(y_1y.dropna()) > 200:
        rf_1y = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_1y.fit(X_scaled[:-252], y_1y.iloc[:-252])
        prob_rf_1y = rf_1y.predict_proba(X_scaled[-1:]) [0][1]
        results.append({
            "Model Name": "Random Forest Ensemble",
            "Focus / Horizon": "1 Jaar (Middellang)",
            "Bullish / Bearish": "BULLISH" if prob_rf_1y > 0.5 else "BEARISH",
            "Stijgingskans (%)": f"{prob_rf_1y * 100:.1f}%"
        })

    # Model 7: Gradient Boosting Ensemble (2 Jaar Horizon)
    y_2y = df['Target_2Y']
    if len(y_2y.dropna()) > 400:
        gb_2y = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_2y.fit(X_scaled[:-504], y_2y.iloc[:-504])
        prob_gb_2y = gb_2y.predict_proba(X_scaled[-1:]) [0][1]
        results.append({
            "Model Name": "Gradient Boosting Ensemble",
            "Focus / Horizon": "2 Jaar (Langetermijn)",
            "Bullish / Bearish": "BULLISH" if prob_gb_2y > 0.5 else "BEARISH",
            "Stijgingskans (%)": f"{prob_gb_2y * 100:.1f}%"
        })

    # Model 8: Support Vector Machine (SVM Trend Classifier)
    svm = SVC(probability=True, kernel='rbf')
    svm.fit(X_scaled[:-5], y_1w.iloc[:-5])
    prob_svm = svm.predict_proba(X_scaled[-1:]) [0][1]
    results.append({
        "Model Name": "Support Vector Machine (SVM)",
        "Focus / Horizon": "1-2 Weken (Trend)",
        "Bullish / Bearish": "BULLISH" if prob_svm > 0.5 else "BEARISH",
        "Stijgingskans (%)": f"{prob_svm * 100:.1f}%"
    })

    # Resultaten omzetten naar DataFrame en afdrukken
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Totale AI Consensus Berekenen (Gewogen naar korte termijn)
    korte_termijn_avg = np.mean([prob_lstm, prob_xgb, prob_lgb, prob_mlp, prob_lr, prob_svm]) * 100
    print("\n-------------------------------------------------------------")
    print(f"TOTALE AI KORTE TERMIJN CONSENSUS (1 WLEK): {korte_termijn_avg:.1f}% STIJGINGSKANS")
    print("SIGNAAL: " + ("STERN BULLISH (KOPEN)" if korte_termijn_avg > 60 else "NEUTRAAL / WATCH" if korte_termijn_avg >= 45 else "BEARISH (VERKOPEN)"))
    print("-------------------------------------------------------------\n")

if __name__ == "__main__":
    # Testen op bijvoorbeeld NVDA, AAPL, MSFT of TSLA
    analyze_stock_ai("NVDA")
