import matplotlib.pyplot as plt
import numpy as np
from data.data_stock import get_stock_data

def visualize_data(data, company_list, features):
    for c in company_list:
        figs, axs = plt.subplots(len(features), 1, figsize=(20, 6*len(features)))
        if len(features) == 1:
            axs = [axs]
        for k, f in enumerate(features):
            axs[k].plot(data[f, c], label=f"{f}", color="white", linewidth=2)
            axs[k].legend(framealpha=1, fontsize=20)
            axs[k].set_title(f"{f}", color="white", fontsize=20)
            axs[k].set_facecolor("#241f1f")  
            axs[k].tick_params(axis='x', colors='white') 
            axs[k].tick_params(axis='y', colors='white')  
            axs[k].spines['bottom'].set_color('white')  
            axs[k].spines['top'].set_color('white')  
            axs[k].spines['left'].set_color('white') 
            axs[k].spines['right'].set_color('white') 
            axs[k].grid(True)
        
        figs.suptitle(f"Stock Data Visualization for the Stock '{c}'", fontsize=26, color="white")
        plt.tight_layout()
        figs.patch.set_facecolor("black")
        plt.show()
        
    