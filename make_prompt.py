import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator

df = pd.read_csv("stock_raw_data.csv")

def add_technical_indicators(df):
    df = df.copy()
    result = []
    for ticker in df['Ticker'].unique():
        temp = df[df['Ticker'] == ticker].copy()
        temp['SMA_10'] = SMAIndicator(temp['Close'], 10).sma_indicator()
        temp['EMA_10'] = EMAIndicator(temp['Close'], 10).ema_indicator()
        temp['RSI_14'] = RSIIndicator(temp['Close'], 14).rsi()
        macd = MACD(temp['Close'])
        temp['MACD'] = macd.macd()
        temp['Return'] = temp['Close'].pct_change().fillna(0)
        result.append(temp)
    return pd.concat(result)

df = add_technical_indicators(df)
def add_bayesian_indicators(df):
    df = df.copy()
    df['Posterior_Trend'] = 0.5
    df['Posterior_Up'] = 0.5
    for ticker in df['Ticker'].unique():
        ticker_data = df[df['Ticker'] == ticker]
        posterior = []
        posterior_up = []
        returns = ticker_data['Return'].values
        for i in range(len(returns)):
            window = returns[max(0,i-9):i+1]
            pos_days = np.sum(window > 0)
            post_trend = (pos_days + 1)/(len(window)+2)
            posterior.append(post_trend)
            if abs(returns[i])>0.02:
                prior = np.mean(posterior[max(0,i-5):i])
                likelihood = 0.8 if returns[i]>0 else 0.2
                post_up_val = (likelihood*prior)/(likelihood*prior + (1-likelihood)*(1-prior))
            else:
                post_up_val = post_trend
            posterior_up.append(post_up_val)
        df.loc[df['Ticker']==ticker,'Posterior_Trend'] = posterior
        df.loc[df['Ticker']==ticker,'Posterior_Up'] = posterior_up
    return df

df = add_bayesian_indicators(df)

df = df.fillna(0) 
df = df.head(20) 

def make_prompt(row):
    return f"Stock: {row['Ticker']}\\nDate: {row['Date']}\\nClose Price: {row['Close']}\\n\\nTechnical Indicators:\\nSMA_10={row['SMA_10']}, EMA_10={row['EMA_10']}, RSI_14={row['RSI_14']}, MACD={row['MACD']}\\n\\nBayesian Indicators:\\nPosterior_Trend={row['Posterior_Trend']}, Posterior_Up={row['Posterior_Up']}\\n\\nQuestion:\\nExplain why the stock might move based on this data in simple terms."

df['Prompt'] = df.apply(make_prompt, axis=1)

df[['Prompt']].to_csv("llm_prompts.csv", index=False, quoting=1)

print("Prompts generated in llm_prompts.csv!")