import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- 1. Define Your Dummy Portfolio ---
PORTFOLIO = [
    {'ticker': 'HDFCBANK.NS', 'quantity': 10},
    {'ticker': 'ICICIBANK.NS', 'quantity': 20},
    {'ticker': 'SBIN.NS', 'quantity': 50},
    {'ticker': 'BAJFINANCE.NS', 'quantity': 5},
    {'ticker': 'TCS.NS', 'quantity': 10},
    {'ticker': 'INFY.NS', 'quantity': 15},
    {'ticker': 'RELIANCE.NS', 'quantity': 8},
    {'ticker': 'HINDUNILVR.NS', 'quantity': 12},
    {'ticker': 'ITC.NS', 'quantity': 100},
    {'ticker': 'MARUTI.NS', 'quantity': 3},
    {'ticker': 'SUNPHARMA.NS', 'quantity': 30},
    {'ticker': 'TATASTEEL.NS', 'quantity': 40},
    {'ticker': 'BHARTIARTL.NS', 'quantity': 25},
    {'ticker': 'LT.NS', 'quantity': 10},
    {'ticker': 'NTPC.NS', 'quantity': 75}
]

# --- 2. Set Up the Streamlit Page ---
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
st.title("My AI-Powered Portfolio Dashboard")

# --- 3. Data Fetching Functions (for Portfolio Tab) ---
@st.cache_data
def get_portfolio_data(portfolio):
    """Fetches detailed info (name, sector, price) for all stocks in the portfolio."""
    data_list = []
    all_tickers = [item['ticker'] for item in portfolio]
    
    try:
        price_data = yf.download(all_tickers, period="2d")
        if price_data.empty:
            st.error("Could not download price data.")
            return pd.DataFrame()
        latest_prices = price_data['Close'].iloc[-1]
    except Exception as e:
        st.error(f"Error downloading price data: {e}")
        return pd.DataFrame()

    st.write("Fetching details (name, sector) for each stock...")
    progress_bar = st.progress(0)
    
    for i, item in enumerate(portfolio):
        ticker_str = item['ticker']
        quantity = item['quantity']
        try:
            ticker_info = yf.Ticker(ticker_str).info
            name = ticker_info.get('longName', ticker_str)
            sector = ticker_info.get('sector', 'Unknown')
            price = latest_prices[ticker_str]
            if pd.isna(price):
                price = ticker_info.get('regularMarketPrice', ticker_info.get('previousClose', 0))
            total_amount = price * quantity
            
            data_list.append({
                'Ticker': ticker_str, 'Name': name, 'Sector': sector,
                'Price': price, 'Quantity': quantity, 'Total Amount': total_amount
            })
        except Exception:
            data_list.append({
                'Ticker': ticker_str, 'Name': 'Error', 'Sector': 'Unknown',
                'Price': 0, 'Quantity': quantity, 'Total Amount': 0
            })
        progress_bar.progress((i + 1) / len(portfolio))
        
    progress_bar.empty()
    return pd.DataFrame(data_list)

@st.cache_data
def get_stock_history(ticker, period):
    """Fetches historical price data for a single stock."""
    try:
        hist_data = yf.download(ticker, period=period)
        if isinstance(hist_data.columns, pd.MultiIndex):
            hist_data.columns = [col[0] for col in hist_data.columns]
        hist_data.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'volume': 'Volume', 'adj close': 'Adj Close'
        }, inplace=True)
        return hist_data
    except Exception as e:
        st.error(f"Error fetching history for {ticker}: {e}")
        return pd.DataFrame()

# --- 4. ML Analysis Function (for ML Tab) ---
@st.cache_data
def run_ml_analysis(ticker):
    """
    Runs the entire ML classifier pipeline for a given stock ticker.
    Returns the prediction and all visuals.
    """
    st.write(f"Running analysis for **{ticker}**...")
    
    # --- 1. Data Collection ---
    st.write("Step 1: Downloading data...")
    try:
        data = yf.download(ticker, start='2020-01-01', end='2025-10-31')
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
            'volume': 'Volume', 'adj close': 'Adj Close'
        }, inplace=True)
        if data.empty:
            st.error("Error: No data downloaded.")
            return None
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

    # --- 2. Feature Engineering ---
    st.write("Step 2: Creating technical indicators...")
    try:
        data.ta.rsi(length=14, append=True)
        data.ta.sma(length=20, append=True) 
        data.ta.sma(length=50, append=True)
        data.ta.bbands(length=20, append=True)
        data.ta.atr(length=14, append=True)
        data.ta.obv(append=True)
        data['SMA_20_vs_50'] = np.where(data['SMA_20'] > data['SMA_50'], 1, 0)
        data = data.dropna()
    except Exception as e:
        st.error(f"Error during feature engineering: {e}")
        return None

    # --- 3. Target Definition ---
    st.write("Step 3: Defining target variable...")
    future_window = 10
    sell_threshold = -0.05
    data['Future_Low'] = data['Low'].shift(-future_window).rolling(window=future_window).min()
    data['Future_Change'] = (data['Future_Low'] - data['Close']) / data['Close']
    data['Target'] = np.where(data['Future_Change'] < sell_threshold, 1, 0)
    data = data.dropna()

    # --- 3a. Data Visualization (for expander) ---
    plots = {}
    sell_signals = data[data['Target'] == 1]
    
    fig1 = plt.figure(figsize=(15, 5))
    plt.plot(data['Close'], label=f'{ticker} Close Price')
    plt.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', label='Sell Signal (Target=1)', s=50)
    plt.title(f'{ticker} Close Price with Future "Sell" Signals')
    plt.legend()
    plots['price_vs_signals'] = fig1
    
    fig2 = plt.figure(figsize=(15, 4))
    plt.plot(data['RSI_14'], label='RSI_14')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title('RSI Indicator')
    plt.legend()
    plots['rsi'] = fig2

    fig3 = plt.figure(figsize=(15, 5))
    plt.plot(data['Close'], label='Close Price', alpha=0.5)
    plt.plot(data['SMA_20'], label='SMA_20 (Fast)')
    plt.plot(data['SMA_50'], label='SMA_50 (Slow)')
    plt.title('Simple Moving Average Crossover')
    plt.legend()
    plots['sma_crossover'] = fig3

    # --- 4. Data Preparation ---
    st.write("Step 4: Preparing data for model...")
    feature_names = [
        'RSI_14', 'SMA_20', 'SMA_50', 'BBL_20_2.0', 'BBU_20_2.0',
        'ATRr_14', 'OBV', 'SMA_20_vs_50'
    ]
    valid_features = [f for f in feature_names if f in data.columns]
    X = data[valid_features]
    y = data['Target']
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- 5. Model Training ---
    st.write("Step 5: Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # --- 6. Model Evaluation (for expander) ---
    st.write("Step 6: Evaluating model...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Keep (0)', 'Sell (1)'])
    
    importances = pd.Series(model.feature_importances_, index=valid_features)
    fig4 = plt.figure(figsize=(10, 5))
    importances.sort_values().plot(kind='barh')
    plt.title('Feature Importances')
    plots['importances'] = fig4

    # --- 7. How to Predict on New Data ---
    st.write("Step 7: Making final prediction...")
    try:
        latest_data_row = X.iloc[[-1]] 
        scaled_latest_data = scaler.transform(latest_data_row)
        prediction = model.predict(scaled_latest_data)
        prediction_proba = model.predict_proba(scaled_latest_data)
        action = "Sell (1)" if prediction[0] == 1 else "Keep (0)"
        confidence = prediction_proba[0][prediction[0]]
        
        return {
            'action': action,
            'confidence': confidence,
            'report': report,
            'plots': plots
        }
    except Exception as e:
        st.error(f"Could not make new prediction: {e}")
        return None

# --- 5. Main App Layout ---
if st.button("Refresh All Data"):
    st.cache_data.clear()

# Create tabs
tab1, tab2 = st.tabs(["📈 Portfolio Dashboard", "🤖 ML Hold/Sell Analysis"])

# --- TAB 1: PORTFOLIO DASHBOARD ---
with tab1:
    df = get_portfolio_data(PORTFOLIO)
    if not df.empty:
        # --- Section 1: Portfolio Holdings ---
        st.subheader("📊 Portfolio Holdings")
        df_display = df.style.format({
            'Price': '₹{:.2f}', 'Total Amount': '₹{:,.2f}'
        })
        st.dataframe(df_display, use_container_width=True)
        
        grand_total = df['Total Amount'].sum()
        st.metric(label="**Total Portfolio Value**", value=f"**₹{grand_total:,.2f}**")
        st.markdown("---")

        # --- Section 2: Interactive Drill-Down Analysis ---
        st.subheader("📈 Interactive Portfolio Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Sector-wise Distribution")
            sector_data = df.groupby('Sector')['Total Amount'].sum().reset_index()
            fig_sector = px.pie(
                sector_data, names='Sector', values='Total Amount', title='Portfolio by Sector'
            )
            fig_sector.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_sector, use_container_width=True)
            
            sector_list = sector_data['Sector'].unique().tolist()
            selected_sector = st.selectbox("Select a Sector to Drill Down", sector_list)

        with col2:
            st.markdown("### Stock Distribution in Selected Sector")
            sector_stocks = df[df['Sector'] == selected_sector]
            fig_stock = px.pie(
                sector_stocks, names='Ticker', values='Total Amount', title=f'Stocks in {selected_sector}'
            )
            fig_stock.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_stock, use_container_width=True)

        st.markdown("---")

        # --- Section 3: Stock Price Chart ---
        st.subheader("💹 Stock Price Chart")
        col_stock, col_period = st.columns([2, 1])
        with col_stock:
            stock_list = sector_stocks['Ticker'].tolist()
            selected_stock = st.selectbox("Select Stock", stock_list)
        with col_period:
            period = st.radio(
                "Select Period", ['1d', '5d', '1mo', '6mo', '1y'], horizontal=True
            )
        
        if selected_stock:
            st.write(f"Displaying {period} history for **{selected_stock}**")
            history = get_stock_history(selected_stock, period)
            if not history.empty:
                fig_price = px.line(history, y='Close', title=f"{selected_stock} Price History")
                st.plotly_chart(fig_price, use_container_width=True)
            else:
                st.warning(f"No price history found for {selected_stock} for the period {period}.")
    else:
        st.error("Could not load portfolio data. Please try again.")

# --- TAB 2: ML HOLD/SELL ANALYSIS ---
with tab2:
    st.subheader("Machine Learning 'Hold vs. Sell' Analysis")
    st.write("Select a stock from your portfolio to run a full ML analysis.")
    
    # Get just the ticker names from the portfolio
    portfolio_tickers = [item['ticker'] for item in PORTFOLIO]
    
    # Create the selectbox for the ML analysis
    ml_stock_choice = st.selectbox("Select a stock to analyze", portfolio_tickers, key="ml_stock")
    
    if ml_stock_choice:
        # Run the full analysis
        analysis_results = run_ml_analysis(ml_stock_choice)
        
        if analysis_results:
            st.markdown("---")
            # Display the final prediction prominently
            st.metric(
                label=f"Prediction for **{ml_stock_choice}**",
                value=analysis_results['action'],
                help=f"Model Confidence: {analysis_results['confidence']:.2f}"
            )
            st.progress(analysis_results['confidence'])
            
            # Put all the training details in an expander
            with st.expander("Show Model Training Details and Evaluation"):
                
                st.subheader("Price with Sell Signals")
                st.pyplot(analysis_results['plots']['price_vs_signals'])
                
                st.subheader("RSI Indicator")
                st.pyplot(analysis_results['plots']['rsi'])
                
                st.subheader("SMA Crossover")
                st.pyplot(analysis_results['plots']['sma_crossover'])
                
                st.subheader("Model Classification Report")
                st.text(analysis_results['report'])
                
                st.subheader("Feature Importances")
                st.pyplot(analysis_results['plots']['importances'])

st.info("Disclaimer: This is an educational project. Data is from Yahoo Finance and may not be real-time. Do not use for real-world trading.")