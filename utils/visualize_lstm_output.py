import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def visualize_lstm_output(model, data, company_list, features, scaler , X_test, Y_test, Y_train, history, scale_data):
    Y_pred = model.predict(X_test)
    
    # inverse transform the the scaled predictions
    if scale_data:
        Y_pred = scaler.inverse_transform(Y_pred)
        Y_test = scaler.inverse_transform(Y_test)

    for i in company_list:
        fig, axs = plt.subplots(len(features),1,figsize=(20,6*len(features)))
        if len(features) == 1:
            axs = [axs]
        for f,j in zip(features,range(len(features))):
            axs[j].plot(Y_test[:,j],label=f"'{f}' Actual Data",linestyle="dashed", color="white",linewidth = 2)
            axs[j].plot(Y_pred[:,j],label=f"'{f}' Prediction Data",linestyle="solid", color="red",linewidth = 2)
            
            axs[j].legend(framealpha = 1, fontsize=14)
            axs[j].set_title(f"{f}",color="white",fontsize=20)
            axs[j].set_facecolor("#241f1f")  
            axs[j].tick_params(axis='x', colors='white') 
            axs[j].tick_params(axis='y', colors='white')  
            axs[j].spines['bottom'].set_color('white')  
            axs[j].spines['top'].set_color('white')  
            axs[j].spines['left'].set_color('white') 
            axs[j].spines['right'].set_color('white') 
            axs[j].legend(framealpha = 1, fontsize=14)
            axs[j].grid(True)
            
        fig.suptitle(f"Stock Data Prediction Results '{i}'", fontsize=26, color="white")
        plt.tight_layout()
        fig.patch.set_facecolor("black")
        plt.savefig("deneme.pdf")
        plt.show()
        
    
    # Plot loss curve
    plt.figure(figsize=(10,5))
    plt.plot(np.array(history.history['loss'])[:], label='Training Loss')
    plt.plot(np.array(history.history['val_loss'])[:], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    