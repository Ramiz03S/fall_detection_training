"""
Iterates over each subjects trials. For each it first performs downsampling for 200 Hz to 100 Hz. 
Then for trials containing falls, it extends the fall phase by 50 samples (0.5 seconds). 
Then a sliding window is performed over the trial's data of length 50 samples and stride 10 samples. 
Each windows label is determined as FALL if it has 40% or more of its samples annotated 
with 1 (fall), otherwise its ADL. The windows and labels generated across all the trials of each subjects are concatenated together.
Resulting in windows of shape (n, 6, 50) and labels (n, ), where n is the total number of windows for each subject.
Windows are then reshapes to (n, 6, 50, 1) and labels into one shot encoded labels with shape (n, 2) since we have two classification categories.
The tuple of (windows, labels) for each subject is saved as an .npz file with name {subject}.npz under a new path.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from collections import Counter
from keras.utils import to_categorical


SubjectAdultList = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
SubjectElderlyList = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]

if __name__ == "__main__":
    
    load_dotenv()  
    
    sisfall_dataset_loaded_path = Path(os.environ.get("SISFALL_DATASET_LOADED_ROOT"))
    sisfall_dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))
    
    sisfall_dataset_processed_path.mkdir(parents=True, exist_ok=True)

    for subject in SubjectAdultList + SubjectElderlyList:
    
        
        subject_windows = []
        subject_labels = []
        
        #for file in subject_files:
        for file in (sisfall_dataset_loaded_path/subject).iterdir():
            
            trial_df = pd.read_csv(sisfall_dataset_loaded_path/subject/file.name)
            
            # resampling down to 100 Hz
            trial_df.index = pd.to_timedelta(np.arange(len(trial_df)) / 200, unit="s")
            
            df1 = trial_df.iloc[:, [0,1,2,3,4,5]].resample("10ms").mean()
            df2 = trial_df.iloc[:, [6]].resample("10ms").max()
            trial_df = pd.concat([df1, df2],axis=1)

            # extending the fall phase by 0.5s at 100 Hz (50 samples)
            trial_np = trial_df.to_numpy()
            if file.name.startswith("F"):
        
                labels = trial_np[:, -1]
                fall_indicies = np.where(labels == 1)
                last_fall_index = fall_indicies[0][-1]
                labels[last_fall_index + 1: last_fall_index + 50 + 1] = 1
                
            # perform sliding windows of length 50 samples, and stride 10 samples
            windows_view = sliding_window_view(trial_np, 50, axis=0)[::10]

            window_labels = np.zeros(windows_view.shape[0]) # each window gets a final class label of 0 or 1 (fall)
            
            for i, labels in enumerate(windows_view[:,-1]):
                
                counter = Counter(labels)
                if counter[1]>= 20: 
                    window_labels[i] = 1
            
            subject_windows.append(windows_view[:,:-1].astype(np.float32))
            subject_labels.append(window_labels.astype(np.int32))
        
        subject_windows = np.concatenate((subject_windows)).reshape((-1, 6, 50 , 1)) # shape (n, 6, 50 , 1)
        subject_labels = to_categorical(np.concatenate((subject_labels)), num_classes=2) # shape (n, 1)
        
        np.savez(sisfall_dataset_processed_path/f"{subject}.npz", windows=subject_windows, labels=subject_labels)