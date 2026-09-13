import os
import pandas as pd

"""
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

"""
ADXL345_factor = (2 * 16) / (2 ** 13)
ITG3200_factor = (2 * 2000) / (2 ** 16)


sisfall_dataset_path = "PC\\datasets\\SisFall_dataset\\"
sisfall_dataset_loaded_path = "PC\\datasets\\SisFall_dataset_loaded\\"
sisfall_dataset_annotated_path = "PC\\datasets\\SisFall_temporally_annotated\\"


SubjectAdultList = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
SubjectElderlyList = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]

Columns_ADXL345 = ["ADXL345_X", "ADXL345_Y", "ADXL345_Z"]
Columns_ITG3200 = ["ITG3200_X", "ITG3200_Y", "ITG3200_Z"]
Columns_MMA8451Q = ["MMA8451Q_X", "MMA8451Q_Y", "MMA8451Q_Z"]

Columns = Columns_ADXL345 + Columns_ITG3200 + Columns_MMA8451Q

for subject in SubjectAdultList + SubjectElderlyList:
    
    subject_files = [f for f in os.listdir(sisfall_dataset_path + subject)]
    # make new dir to store the loaded dataset
    os.makedirs(sisfall_dataset_loaded_path + subject, exist_ok=True) 
    
    for file in subject_files:
        
        if file.endswith(".txt"): # so that pd.read_csv skips non text files
            
            sensor_df = pd.read_csv(sisfall_dataset_path + subject + "\\" + file, names=Columns)
            sensor_df = sensor_df.iloc[:, [0,1,2,3,4,5]]
            # TODO: assumed that the ADXL345 accelerometer was used
            label_df = pd.read_csv(sisfall_dataset_annotated_path + subject + "\\" + file, names=["Label"])
            trial_df = pd.concat([sensor_df, label_df], axis=1)
            
            # Conversion from bits into g and deg/sec units
            trial_df[Columns_ADXL345] = trial_df[Columns_ADXL345] * ADXL345_factor
            trial_df[Columns_ITG3200] = trial_df[Columns_ITG3200] * ITG3200_factor
            # replacing the ALERT into BKG as advised by https://doi.org/10.1109/TETC.2020.3027454
            # trial_df["Label"] = trial_df["Label"].apply(lambda x: 0 if x == 1 else x)
            trial_df["Label"] = trial_df["Label"].replace(1,0)
            
            trial_df.to_csv(sisfall_dataset_loaded_path + subject + "\\" + file, index=False)
        
        
    
