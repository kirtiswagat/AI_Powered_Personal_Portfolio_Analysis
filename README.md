
-----

# AI-Powered Portfolio Dashboard

This is a Streamlit web application that provides a comprehensive dashboard for a personal stock portfolio. It combines portfolio tracking, interactive visualizations, and an AI-powered "Hold vs. Sell" analysis for individual stocks.

This application is split into two main sections:

1.  **📈 Portfolio Dashboard:** An interactive view of your holdings, portfolio value, and sector-wise allocation.
2.  **🤖 ML Hold/Sell Analysis:** A tool to run an on-demand machine learning analysis on any stock in your portfolio to get a simple "Keep" or "Sell" signal.

## 🚀 Key Features

  * **Portfolio Tracking:** View all your holdings, quantities, current prices, and total value in a clean, filterable table.
  * **Total Value Metric:** See the real-time total value of your entire portfolio.
  * **Interactive Pie Charts:** Instantly visualize your portfolio's allocation by sector.
  * **Drill-Down Analysis:** Select a sector from the main pie chart to see a second chart showing the stock distribution *within* that sector.
  * **Historical Price Charts:** Select any stock from your drill-down view to instantly plot its historical price chart for various periods (1 day, 1 month, 1 year, etc.).
  * **On-Demand ML Analysis:** Select any stock from your portfolio to run a full machine learning pipeline.
  * **Automated Feature Engineering:** Automatically generates technical indicators like **RSI, SMAs, Bollinger Bands, ATR,** and **OBV** to be used as features.
  * **AI-Powered Signal:** Trains a Random Forest Classifier to predict whether a stock is a "Keep (0)" or "Sell (1)" based on historical patterns.
  * **Model Explainability:** View the model's performance, classification report, and feature importance plot in an expandable section.

## 🛠️ Installation

1.  Ensure you have Python 3.8 or newer installed.
2.  Clone this repository or download the source code.
3.  Install all required libraries using pip:

<!-- end list -->

```bash
pip install streamlit yfinance pandas plotly pandas-ta scikit-learn matplotlib
```

## 🏃 How to Run

1.  Save the main script as `dashboard.py` (or any name you prefer).
2.  Open your terminal or command prompt.
3.  Navigate to the directory where you saved the script.
4.  Run the following command:

<!-- end list -->

```bash
streamlit run dashboard.py
```

Streamlit will automatically open the application in your default web browser.

## 📄 Code Explanation

The application is built around a few core components:

### 1\. Portfolio Definition (`PORTFOLIO`)

This is a hardcoded Python list of dictionaries at the beginning of the script. It acts as the "database" for the app, defining which stocks you own and in what quantity.

```python
PORTFOLIO = [
    {'ticker': 'HDFCBANK.NS', 'quantity': 10},
    {'ticker': 'ICICIBANK.NS', 'quantity': 20},
    # ... other stocks
]
```

### 2\. Caching & Data Fetching (`@st.cache_data`)

The app uses Streamlit's `@st.cache_data` decorator on its data-fetching functions. This is a critical optimization that "remembers" the results of a function. The app will only re-download data from Yahoo Finance if the input (like a ticker) changes, making the dashboard fast and responsive.

  * `get_portfolio_data()`: Fetches the *current* data (name, sector, price) for all stocks at once to populate the main dashboard.
  * `get_stock_history()`: Fetches the *historical* price data for a *single* stock, used for the line charts.

Both functions also include a vital fix to "flatten" the **MultiIndex columns** that `yfinance` returns, making the data usable for `pandas` and `plotly`.

### 3\. Tab 1: Portfolio Dashboard

This tab is the main interface for portfolio tracking.

  * **Holdings Table:** The `get_portfolio_data()` function's DataFrame is displayed using `st.dataframe` with custom currency formatting.
  * **Interactive Drill-Down:** The two pie charts are linked.
    1.  The first chart (`fig_sector`) shows the total value grouped by **Sector**.
    2.  A select box (`st.selectbox`) lets you choose one of those sectors.
    3.  The main DataFrame is filtered based on your selection.
    4.  The second chart (`fig_stock`) is generated from this *filtered* DataFrame, showing the stock breakdown within that one sector.
  * **Stock Price Chart:** This is the final "drill-down." The `selected_stock` selectbox is populated *only* with tickers from the filtered sector. The `get_stock_history()` function is then called to draw the `plotly` line chart.

### 4\. Tab 2: ML Hold/Sell Analysis

This tab contains the entire machine learning pipeline, wrapped in the `run_ml_analysis(ticker)` function.

  * **Step 1-2: Data & Features:** The function downloads several years of historical data for the selected stock. It then uses the `pandas_ta` library to create a rich set of technical indicators (features).
  * **Step 3: Target Definition (The "Secret Sauce"):** This is where the problem becomes a machine learning problem.
      * It creates a "target" variable named `Target`.
      * A day is labeled as **"Sell (1)"** if the stock's price drops by 5% or more at any point in the next 10 trading days.
      * If it doesn't, it's labeled as **"Keep (0)"**.
  * **Step 4-5: Model Training:** The data is split into training and testing sets. A `RandomForestClassifier` is trained on this data. We use `class_weight='balanced'` because "Sell" events are much rarer than "Keep" events, which helps the model pay closer attention to them.
  * **Step 6-7: Prediction:** The model is used to predict a signal for the **single most recent day** of data. This final prediction (`Keep (0)` or `Sell (1)`) is what's displayed to the user.
  * **Results:** The function returns the final prediction, confidence, and all the plots, which are displayed in the Streamlit app. The `st.expander` is used to hide the complex plots and reports, keeping the main UI clean.

## ⚠️ Disclaimer

This is an educational project. All data is provided by Yahoo Finance and may not be real-time. The machine learning model is for illustrative purposes only and is **not financial advice**. Do not use this project for real-world trading.
