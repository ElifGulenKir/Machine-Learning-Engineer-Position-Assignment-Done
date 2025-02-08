import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import datetime as dt



def get_rsi(data, rsi_feature: str = "Open", days: int = 14, company_list: list = None, features: list = None):
    delta = data[[rsi_feature]].diff(1)
    delta.dropna(inplace=True)
    
    positive = delta.copy()
    negative = delta.copy()

    positive[positive < 0] = 0
    negative[negative > 0] = 0
    
    average_gain = positive.rolling(window = days).mean()
    average_loss = abs(negative.rolling(window = days).mean()) 
    
    relative_strength = average_gain / average_loss
    RSI = 100.0 - (100.0 / (1.0 + relative_strength))   
    
    combined = pd.DataFrame()
    
    for i in range(len(company_list)):
        for f in features:
            combined[f] = data[f]
            combined[f'RSI_{company_list[i]}'] = RSI
            
    return combined


def visualize_rsi(combined, rsi_feature, company_list, savefig:bool = False):
    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(211)
    ax1.plot(combined.index, combined[f"{rsi_feature}"], color='lightgray')
    ax1.set_title(f"{rsi_feature}", color='white')
              
    ax1.grid(True, color='#555555') 
    ax1.set_axisbelow(True)
    ax1.set_facecolor('black')         
    ax1.figure.set_facecolor('#121212')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(combined.index, combined[f'RSI_{company_list[0]}'], color='lightgray')
    ax2.axhline(0, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')

    ax2.set_title(f"RSI Value of {rsi_feature}",color="white")
    ax2.grid(False) 
    ax2.set_axisbelow(True)
    ax2.set_facecolor('black')         
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    
    if (savefig):
        plt.savefig(f"results/RSI_{rsi_feature}.pdf")

    plt.show()
    
    
    
    
