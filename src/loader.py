"""
Iterates through all subjects and trial runs of the Sisfalls dataset. For each, it drops the last 3 columns of data 
associated with the second accelerometer. Then it concatenates each time point's 6 sensor values with its temporal annotation.
Following that, the sensors raw data values are converted back into the values with units using the following information form the readme file:

ADXL345:
Resolution: 13 bits
Range: +-16g

ITG3200
Resolution: 16 bits
Range: +-2000 deg/s

In order to convert the acceleration data (AD) given in bits into gravity, use this equation: 
Acceleration [g]: [(2*Range)/(2^Resolution)]*AD
For ADXL345 the multiplication factor is (2 * 16)/(2^13)

In order to convert the rotation data (RD) given in bits into angular velocity, use this equation:
Angular velocity [deg/s]: [(2*Range)/(2^Resolution)]*RD
For ITG3200 the multiplication factor is (2 * 2000)/(2^16)

It then replaces the ALERT (1) temporal labels into BKG (0) as advised by https://doi.org/10.1109/TETC.2020.3027454
and FALL (2) into (1) for later one hot encoding. 
Finally, we save back each trials csv following these changes into another folder following the same structure as
the original dataset.

"""

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm

ADXL345_factor = (2 * 16) / (2 ** 13)
ITG3200_factor = (2 * 2000) / (2 ** 16)

SubjectAdultList = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
SubjectElderlyList = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]

Columns_ADXL345 = ["ADXL345_X", "ADXL345_Y", "ADXL345_Z"]
Columns_ITG3200 = ["ITG3200_X", "ITG3200_Y", "ITG3200_Z"]
Columns_MMA8451Q = ["MMA8451Q_X", "MMA8451Q_Y", "MMA8451Q_Z"]

Columns = Columns_ADXL345 + Columns_ITG3200 + Columns_MMA8451Q

kfall_subject_codes = [
    "SA06", "SA07", "SA08", "SA09", "SA10", "SA11", "SA12", "SA13",
    "SA14", "SA15", "SA16", "SA17", "SA18", "SA19", "SA20", "SA21",
    "SA22", "SA23", "SA24", "SA25", "SA26", "SA27", "SA28", "SA29",
    "SA30", "SA31", "SA32", "SA33", "SA35", "SA36", "SA37", "SA38",
]

def load_kfall():

    load_dotenv()

    kfall_dataset_path = Path(os.environ.get("KFALL_DATASET_ROOT"))
    kfall_dataset_loaded_path = Path(os.environ.get("KFALL_DATASET_LOADED_ROOT"))

    subject_fall_dicts = {}
    for file in (kfall_dataset_path/"label_data").iterdir():
        subject = file.name[0:4]
        (kfall_dataset_loaded_path/subject).mkdir(parents=True, exist_ok=True)

        df = pd.read_excel(file)
        fall_tasks_dicts = {}
        for row in range(len(df)):

            task_code_s = df.iloc[row]["Task Code (Task ID)"]
            if isinstance(task_code_s, str):
                task_code = task_code_s[5:7]
                fall_tasks_dicts[task_code] = {}
        
        
            trial_id = df.iloc[row]["Trial ID"]
            onset_frame = df.iloc[row]["Fall_onset_frame"]
            impact_frame = df.iloc[row]["Fall_impact_frame"]

            fall_tasks_dicts[task_code][str(trial_id)] = {"onset_frame": onset_frame, "impact_frame": impact_frame}
    
        subject_fall_dicts[subject] = {"falls_info": fall_tasks_dicts}

    for subject in tqdm(kfall_subject_codes):
        (kfall_dataset_loaded_path/subject).mkdir(parents=True, exist_ok=True)

        falls_info = subject_fall_dicts[subject]["falls_info"]

        for file in (kfall_dataset_path/"sensor_data"/subject).iterdir():
            file_task_id = file.name[4:6]
            file_trial_id = file.name[8:9]
            file_onset_frame = None
            file_impact_frame = None

            if file_task_id in falls_info.keys():
                file_onset_frame = falls_info[file_task_id][file_trial_id]["onset_frame"]
                file_impact_frame = falls_info[file_task_id][file_trial_id]["impact_frame"]

            trial_df = pd.read_csv(file).iloc[:, [1, 2, 3, 4, 5, 6, 7]]
            
            trial_df["Label"] = 0
            if file_onset_frame is not None and file_impact_frame is not None:
                trial_df.loc[file_onset_frame:file_impact_frame, "Label"] = 1
            
            trial_df.to_csv(kfall_dataset_loaded_path/subject/file.name, index=False)

    




def load_sisfall():
    
    load_dotenv()  
    
    sisfall_dataset_path = Path(os.environ.get("SISFALL_DATASET_ROOT"))
    sisfall_dataset_annotated_path = Path(os.environ.get("SISFALL_TEMPORAL_ANNOTATIONS_ROOT"))
    sisfall_dataset_loaded_path = Path(os.environ.get("SISFALL_DATASET_LOADED_ROOT"))
    
    for subject in tqdm(SubjectAdultList + SubjectElderlyList):
        
        (sisfall_dataset_loaded_path/subject).mkdir(parents=True, exist_ok=True)
        
        for file in (sisfall_dataset_path/subject).iterdir():
            
            if file.suffix == '.txt': # so that pd.read_csv skips non text files
                
                sensor_df = pd.read_csv(file, names=Columns)
                # TODO: assumed that the ADXL345 accelerometer was used
                sensor_df = sensor_df.iloc[:, [0,1,2,3,4,5]]
                label_df = pd.read_csv(sisfall_dataset_annotated_path/subject/file.name, names=["Label"])

                if len(sensor_df) != len(label_df):
                    diff = len(label_df) - len(sensor_df)
                    print(f"{subject}/{file.name}: labels - sensors = {diff}")
                    n = min(len(sensor_df), len(label_df))
                    sensor_df, label_df = sensor_df.iloc[:n], label_df.iloc[:n]
                    
                trial_df = pd.concat([sensor_df, label_df], axis=1)
                
                # Conversion from bits into g and deg/sec units
                trial_df[Columns_ADXL345] = trial_df[Columns_ADXL345] * ADXL345_factor
                trial_df[Columns_ITG3200] = trial_df[Columns_ITG3200] * ITG3200_factor
                
                # replacing the ALERT into BKG as advised by https://doi.org/10.1109/TETC.2020.3027454
                trial_df["Label"] = trial_df["Label"].replace(1,0)
                # Labelling Falls as 1
                trial_df["Label"] = trial_df["Label"].replace(2,1)
                
                
                trial_df.to_csv(sisfall_dataset_loaded_path/subject/file.name, index=False)

if __name__ == "__main__":
    load_kfall()
    # load_sisfall()