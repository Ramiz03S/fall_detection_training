import os
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from collections import Counter


sisfall_dataset_loaded_path = "PC\\datasets\\SisFall_dataset_loaded\\"
sisfall_dataset_processed_path = "PC\\datasets\\SisFall_dataset_processed\\"

SubjectAdultList = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
SubjectElderlyList = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]

for subject in SubjectAdultList + SubjectElderlyList:
    
    os.makedirs(sisfall_dataset_processed_path + subject, exist_ok=True)
    subject_files = [f for f in os.listdir(sisfall_dataset_loaded_path + subject)]
    
    for file in subject_files:
        
        trial_df = pd.read_csv(sisfall_dataset_loaded_path + subject + "\\" + file)
        
        # resampling down to 100 Hz
        trial_df.index = pd.to_timedelta(np.arange(len(trial_df)) / 200, unit="s")
        
        df1 = trial_df.iloc[:, [0,1,2,3,4,5]].resample("10ms").mean()
        df2 = trial_df.iloc[:, [6]].resample("10ms").max()
        trial_df = pd.concat([df1, df2],axis=1)

        # extending the fall phase by 0.5s at 100 Hz (50 samples)
        trial_np = trial_df.to_numpy()
        if file.startswith("F"):
    
            labels = trial_np[:, -1]
            fall_indicies = np.where(labels == 2)
            last_fall_index = fall_indicies[0][-1]
            labels[last_fall_index + 1: last_fall_index + 50 + 1] = 2
            
        # perform sliding windows of length 50 samples, and stride 10 samples
        windows_view = sliding_window_view(trial_np, 50, axis=0)[::10]

        window_labels = np.zeros(windows_view.shape[0]) # each window gets a final class label of 0 or 2 (fall)
        for i, labels in enumerate(windows_view[:,-1]):
            
            counter = Counter(labels)
            if counter[2]>= 20: 
                window_labels[i] = 2
        
        np.savez(sisfall_dataset_processed_path + subject + "\\" + file[:-4], windows=windows_view, labels=window_labels)
        
        
        
        
        
    
