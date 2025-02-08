import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def bollinger_bands(data, feature: str = "Open", window: int =20, num_std: float =2):
    data = data[feature].squeeze()

    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    bollinger_df = pd.DataFrame({
        f'{feature}': data,
        'Rolling Mean': rolling_mean,
        'Upper Band': upper_band,
        'Lower Band': lower_band
    }, index=data.index)  # ✅ Correctly setting index

    return bollinger_df


def visualize_bollinger_bands(data, feature):
    plt.figure(figsize=(12, 6))
    
    plt.plot(data.index, data[f'{feature}'], label=f'{feature}', color='blue', linewidth=1)
    plt.plot(data.index, data['Rolling Mean'], label='Rolling Mean', color='orange', linestyle='dashed')
    plt.plot(data.index, data['Upper Band'], label='Upper Band', color='green', linestyle='dotted')
    plt.plot(data.index, data['Lower Band'], label='Lower Band', color='red', linestyle='dotted')

    plt.fill_between(data.index, data['Upper Band'], data['Lower Band'], color='gray', alpha=0.2)

    plt.title('Bollinger Bands')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()