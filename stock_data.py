import yfinance as yf
import pandas as pd
tickers = ["AAPL", "MSFT", "TSLA", "JPM", "AMZN", "XOM"]

start_date = "2022-01-01"
end_date = "2023-12-31"

all_data = []
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    data.columns = data.columns.get_level_values(0)

    data['Ticker'] = ticker
    data.reset_index(inplace=True)
    all_data.append(data)

df = pd.concat(all_data)
df.to_csv("stock_raw_data.csv", index=False)
print("Stock raw data saved!")

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

def add_technical_indicators(df):
    df = df.copy()
    result = []

    for ticker in df['Ticker'].unique():
        temp = df[df['Ticker'] == ticker].copy()

        temp['SMA_10'] = SMAIndicator(temp['Close'], 10).sma_indicator()
        temp['SMA_20'] = SMAIndicator(temp['Close'], 20).sma_indicator()
        temp['SMA_50'] = SMAIndicator(temp['Close'], 50).sma_indicator()

        temp['EMA_10'] = EMAIndicator(temp['Close'], 10).ema_indicator()
        temp['EMA_20'] = EMAIndicator(temp['Close'], 20).ema_indicator()

        temp['RSI_14'] = RSIIndicator(temp['Close'], 14).rsi()

        macd = MACD(temp['Close'])
        temp['MACD'] = macd.macd()
        temp['MACD_signal'] = macd.macd_signal()
        temp['MACD_hist'] = macd.macd_diff()

        temp['Return'] = temp['Close'].pct_change()
        temp['Volatility_10d'] = temp['Return'].rolling(10).std()

        result.append(temp)

    return pd.concat(result)

df = add_technical_indicators(df)
df.to_csv("stock_with_technical_indicators.csv", index=False)
print("Technical indicators added!")

import numpy as np

def add_bayesian_indicators(df):
    df = df.copy()
    df['Posterior_Trend'] = 0.5 
    df['Prior_Up'] = 0.5
    df['Posterior_Up'] = 0.5
    df['Is_Large_Move'] = False
    
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker].copy()
        posterior = []
        prior_up = []
        posterior_up = []
        large_move = []
        returns = ticker_data['Return'].fillna(0).values
        
        for i in range(len(returns)):
            window = returns[max(0, i-9):i+1]
            pos_days = np.sum(window > 0)
            posterior_trend = (pos_days + 1) / (len(window) + 2) 
            posterior.append(posterior_trend)
            if abs(returns[i]) > 0.02:
                is_event = True
                prior = np.mean(posterior[max(0, i-5):i]) 
                likelihood = 0.8 if returns[i] > 0 else 0.2
                post_up = (likelihood * prior) / (likelihood * prior + (1-likelihood)*(1-prior))
            else:
                is_event = False
                prior = posterior[i]
                post_up = prior
            
            prior_up.append(prior)
            posterior_up.append(post_up)
            large_move.append(is_event)
        
        df.loc[df['Ticker'] == ticker, 'Posterior_Trend'] = posterior
        df.loc[df['Ticker'] == ticker, 'Prior_Up'] = prior_up
        df.loc[df['Ticker'] == ticker, 'Posterior_Up'] = posterior_up
        df.loc[df['Ticker'] == ticker, 'Is_Large_Move'] = large_move
        
    return df

df = add_bayesian_indicators(df)
df.to_csv("stock_with_technical_bayesian.csv", index=False)
print("Bayesian indicators added!")

import numpy as np

def add_bayesian_indicators(df):
    df = df.copy()
    df['Posterior_Trend'] = 0.5  # 默认初始概率
    df['Prior_Up'] = 0.5
    df['Posterior_Up'] = 0.5
    df['Is_Large_Move'] = False
    
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker].copy()
        posterior = []
        prior_up = []
        posterior_up = []
        large_move = []
        returns = ticker_data['Return'].fillna(0).values
        
        for i in range(len(returns)):
            # Posterior Trend Probability (过去10天)
            window = returns[max(0, i-9):i+1]
            pos_days = np.sum(window > 0)
            posterior_trend = (pos_days + 1) / (len(window) + 2)  # Laplace smoothing
            posterior.append(posterior_trend)
            
            # Event-Based Update (大涨/大跌)
            if abs(returns[i]) > 0.02:
                is_event = True
                prior = np.mean(posterior[max(0, i-5):i])  # 最近5天平均
                likelihood = 0.8 if returns[i] > 0 else 0.2
                post_up = (likelihood * prior) / (likelihood * prior + (1-likelihood)*(1-prior))
            else:
                is_event = False
                prior = posterior[i]
                post_up = prior
            
            prior_up.append(prior)
            posterior_up.append(post_up)
            large_move.append(is_event)
        
        df.loc[df['Ticker'] == ticker, 'Posterior_Trend'] = posterior
        df.loc[df['Ticker'] == ticker, 'Prior_Up'] = prior_up
        df.loc[df['Ticker'] == ticker, 'Posterior_Up'] = posterior_up
        df.loc[df['Ticker'] == ticker, 'Is_Large_Move'] = large_move
        
    return df

df = add_bayesian_indicators(df)
df.to_csv("stock_with_technical_bayesian.csv", index=False)
print("Bayesian indicators added!")