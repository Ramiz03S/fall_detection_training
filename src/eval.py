import os
import json
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




if __name__ == "__main__":
    
    load_dotenv()  
    runs_path = Path(os.environ.get("RUNS_ROOT"))
    
    runs = [4, 5, 6, 7, 8, 9]
    
    for run in runs:
        with open(runs_path/f"run_{run}/fold_0_hparams.json", "r") as file:
            hparams = json.load(file)
            
        
        df_metrics = pd.read_csv(runs_path/f"run_{run}"/"metrics"/"fold_0.log").iloc[:, [0,2,3,5,6]]
        metrics = df_metrics.to_numpy()
        pass
    
        fig, ax = plt.subplots()

        ax.plot(metrics[:,0], metrics[:,1], 'b', label='train_sensitivity')
        ax.plot(metrics[:,0], metrics[:,2], 'g', label='train_specificity')
        ax.plot(metrics[:,0], metrics[:,3], 'm', label='val_sensitivity')
        ax.plot(metrics[:,0], metrics[:,4], 'r', label='val_specificity')
        ax.legend(loc="lower right")
        ax.set_xlabel('Epochs')
        ax.grid()
        fig.savefig(runs_path/f"run_{run}"/f"fold_0_metrics_plot.png")
    
    