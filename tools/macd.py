import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go

def get_macd(data):
    data = data.copy()  # Prevent modifying original DataFrame

    # Calculate EMAs
    data['EMA12'] = data['Open'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Open'].ewm(span=26, adjust=False).mean()
    
    # Compute MACD Line
    data['MACD'] = data['EMA12'] - data['EMA26']
    
    # Compute Signal Line (9-day EMA of MACD)
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
  
    return data


def visualize_macd(data):
    fig, ax = plt.subplots(2, figsize=(12, 6), sharex=True)
    

    # Check if data is empty
    if data.empty:
        print("Error: No valid MACD values available for plotting.")
        return

    # Plot MACD & Signal Line
    ax[0].plot(data.index, data['MACD'], label='MACD', color='blue')
    ax[0].plot(data.index, data['Signal_Line'], label='Signal Line', color='red')
    ax[0].legend()
    ax[0].set_title("MACD & Signal Line")
    
    # Plot Histogram
    colors = ['green' if val >= 0 else 'red' for val in (data['MACD'] - data['Signal_Line'])]
    ax[1].bar(data.index, data['MACD'] - data['Signal_Line'], label='Histogram', color=colors)

    ax[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax[1].legend()
    ax[1].set_title("MACD Histogram")

    plt.show()