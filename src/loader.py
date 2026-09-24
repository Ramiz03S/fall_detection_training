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



ADXL345_factor = (2 * 16) / (2 ** 13)
ITG3200_factor = (2 * 2000) / (2 ** 16)

SubjectAdultList = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
SubjectElderlyList = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]

Columns_ADXL345 = ["ADXL345_X", "ADXL345_Y", "ADXL345_Z"]
Columns_ITG3200 = ["ITG3200_X", "ITG3200_Y", "ITG3200_Z"]
Columns_MMA8451Q = ["MMA8451Q_X", "MMA8451Q_Y", "MMA8451Q_Z"]

Columns = Columns_ADXL345 + Columns_ITG3200 + Columns_MMA8451Q

if __name__ == "__main__":
    
    load_dotenv()  
    
    sisfall_dataset_path = Path(os.environ.get("SISFALL_DATASET_ROOT"))
    sisfall_dataset_annotated_path = Path(os.environ.get("SISFALL_TEMPORAL_ANNOTATIONS_ROOT"))
    sisfall_dataset_loaded_path = Path(os.environ.get("SISFALL_DATASET_LOADED_ROOT"))
    
    for subject in SubjectAdultList + SubjectElderlyList:
        
        
        (sisfall_dataset_loaded_path/subject).mkdir(parents=True, exist_ok=True)
        
        for file in (sisfall_dataset_path/subject).iterdir():
            
            if file.suffix == '.txt': # so that pd.read_csv skips non text files
                
                sensor_df = pd.read_csv(file, names=Columns)
                sensor_df = sensor_df.iloc[:, [0,1,2,3,4,5]]
                # TODO: assumed that the ADXL345 accelerometer was used
                
                label_df = pd.read_csv(sisfall_dataset_annotated_path/subject/file.name, names=["Label"])
                trial_df = pd.concat([sensor_df, label_df], axis=1)
                
                # Conversion from bits into g and deg/sec units
                trial_df[Columns_ADXL345] = trial_df[Columns_ADXL345] * ADXL345_factor
                trial_df[Columns_ITG3200] = trial_df[Columns_ITG3200] * ITG3200_factor
                
                # replacing the ALERT into BKG as advised by https://doi.org/10.1109/TETC.2020.3027454
                trial_df["Label"] = trial_df["Label"].replace(1,0)
                # Labelling Falls as 1
                trial_df["Label"] = trial_df["Label"].replace(2,1)
                
                
                trial_df.to_csv(sisfall_dataset_loaded_path/subject/file.name, index=False)