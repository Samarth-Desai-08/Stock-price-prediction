# 📈 Stock Price Prediction

This project demonstrates a simple **Stock Price Prediction** pipeline using historical stock market data. It leverages **NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, and Yahoo Finance (yfinance)** to train and evaluate models like **Linear Regression** and **Decision Trees**.

> ⚠️ **Note:** This project is for educational purposes only and **not** financial advice.

---

## 🚀 Features

* Download historical stock data using **yfinance**
* Feature engineering: lag returns, moving averages, volume changes, and price ranges
* Predict next-day closing prices
* Models: **Linear Regression** and **Decision Tree Regressor**
* Evaluation metrics: RMSE, MAE, R²
* Visualization of actual vs predicted stock prices
* Save trained models and preprocessing artifacts

---

## 📂 Project Structure

```
├── stock_price_prediction.py   # Main script
├── requirements.txt            # Python dependencies
├── artifacts/                  # Saved models, scalers, plots
└── README.md                   # Project documentation
```

---

## 🔧 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/stock-price-prediction.git
   cd stock-price-prediction
   ```
2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

Run the script with default arguments (predicts **AAPL**):

```bash
python stock_price_prediction.py
```

Or specify a ticker and date range:

```bash
python stock_price_prediction.py --ticker MSFT --start 2018-01-01 --end 2024-12-31
```

The results will include:

* Model evaluation metrics printed in the console
* A plot of actual vs predicted prices saved in the `artifacts/` folder
* Saved models and preprocessing artifacts (`.pkl` files)

---

## 📊 Example Output

* **Console Output:**

  ```
  Evaluation Results:
                    RMSE    MAE     R2
  LinearRegression  2.345  1.789  0.842
  DecisionTree      3.112  2.456  0.791
  ```

* **Plot:** ![Prediction Plot](artifacts/example_predictions.png)

---

## 📦 Requirements

See `requirements.txt` for the full list of dependencies.

---

## 🛠 Tech Stack

* Python 3.8+
* NumPy, Pandas
* Matplotlib, Seaborn
* scikit-learn
* yfinance
* joblib

---

## 📜 License

This project is licensed under the MIT License. Feel free to use and modify it.
