import os
import json
from pathlib import Path
from dotenv import load_dotenv
from numpy import log
import tensorflow as tf
import keras
from sklearn.model_selection import KFold, StratifiedKFold
from dataset_utils import make_dataset, make_X_Y
from datetime import datetime

RUN = "22_a_0.10_G_2.5_t_0.10_wN_all"
RUN_DESCRIPTION = "Alpha [0.90, 0.10]. On the whole subject set with StratifiedKFold. Gama 2.5 instead of 2. Added threshold of [0.90, 0.10] for metric calculation. Applying norm from  X_train. Saving model on last epoch. Doing one fold"
EPOCHS = 200
BATCH_SIZE = 512
ALPHA_FALLS = 0.10
GAMMA = 2.5
THRESHOLD_FALLS = 0.10
RATE_FALLS = 0.03
LEARNING_RATE = 0.0005
N_SPLITS = 5
RAND_STATE = 5
DATETIME = str(datetime.now().replace(microsecond=0))

output_signature_window = tf.TensorSpec(shape=(6,50,1), dtype=tf.float32)
output_signature_label = tf.TensorSpec(shape=(2,), dtype=tf.int32)
output_signature = (output_signature_window, output_signature_label)

subject_adult_list = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
subject_elderly_list = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]
subject_list = subject_adult_list + subject_elderly_list
# 0 subjects are complete with falls and adl, 1 subjects just have adls
subject_type = [0 for _ in range(1,24)] + [1 for _ in range(1,6)] + [0] + [1 for _ in range(7,16)]

SUBJECTS = subject_list

def make_TinyCNN():
    
    inputs = keras.Input(shape=(6, 50, 1))
    normalization = keras.layers.Normalization(axis=1)(inputs)
    temp_conv = keras.layers.Conv2D(filters=8, kernel_size=(1,16), padding="same")(normalization)
    relu1 = keras.layers.Activation('relu')(temp_conv)
    spatial_conv = keras.layers.Conv2D(filters=8, kernel_size=(6,1), padding="same")(relu1)
    relu2 = keras.layers.Activation('relu')(spatial_conv)
    global_avg_pooling = keras.layers.GlobalAveragePooling2D()(relu2)
    outputs = keras.layers.Dense(units=2, activation="softmax")(global_avg_pooling)
    
    return keras.Model(inputs=inputs, outputs=outputs)

def adapt_normalization_layer_g(tinyCNN_model_instance, train_set, output_signature, batch_size):
    layer = tinyCNN_model_instance.get_layer(index=1)
    layer.adapt(make_dataset(train_set, output_signature=output_signature, batch_size=batch_size, with_labels=False))
    
def adapt_normalization_layer(tinyCNN_model_instance, X_train):
    layer = tinyCNN_model_instance.get_layer(index=1)
    layer.adapt(X_train)
    
def output_bias_init(tinyCNN_model_instance):
    dense_layer = tinyCNN_model_instance.layers[-1]
    dense_layer.bias.assign(log([1 - RATE_FALLS, RATE_FALLS]).astype("float32"))


def compile_model(tinyCNN_model_instance, lr = LEARNING_RATE, alpha_falls = ALPHA_FALLS, gamma = GAMMA):
    tinyCNN_model_instance.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr), 
        jit_compile=False,
        loss=keras.losses.CategoricalFocalCrossentropy(alpha=[1 - alpha_falls, alpha_falls], gamma=gamma),
        metrics=[
            # keras.metrics.CategoricalAccuracy(),
            keras.metrics.Recall(class_id=1, name="sensitivity", thresholds=THRESHOLD_FALLS), 
            # Of the actual falls, how many did the model catch (as falls)? Identifies true positives (falls)
            # miss rate is (1 − sensitivity)
            keras.metrics.Recall(class_id=0, name="specificity", thresholds=1-THRESHOLD_FALLS), 
            # Of the actual negatives (ADL), how many did the model correctly reject (as falls)? Identifies true negatives (ADL)
            # 1 − specificity is the false alarm rate
        ])


if __name__ == "__main__":
    
    devices = tf.config.list_physical_devices('GPU')
    print("number of gpu devices: ", len(devices)) 
    print("printing tf.test.is_built_with_cuda(): ", tf.test.is_built_with_cuda())
    
    load_dotenv()  
    sisfall_dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))
    runs_path = Path(os.environ.get("RUNS_ROOT"))
   
    (runs_path/f'run_{RUN}'/'models').mkdir(parents=True, exist_ok=True)
    (runs_path/f'run_{RUN}'/'metrics').mkdir(parents=True, exist_ok=True)
    (runs_path/f'run_{RUN}'/'tensorboard').mkdir(parents=True, exist_ok=True)
    
    hparams = {
        "date_time": DATETIME,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "alpha": ALPHA_FALLS,
        "gamma": GAMMA,
        "threshold_falls": THRESHOLD_FALLS,
        "rate_falls": RATE_FALLS,
        "n_splits": N_SPLITS,
        "rand_state": RAND_STATE,
        "run_description": RUN_DESCRIPTION,
        "dataset_used": SUBJECTS            
    }
    
    # kf = KFold(n_splits=N_SPLITS, random_state=RAND_STATE, shuffle=True)
    skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RAND_STATE, shuffle=True)

    for fold, (train_index, test_index) in enumerate(skf.split(subject_list, subject_type)):
    # for fold, (train_index, test_index) in enumerate(kf.split(SUBJECTS)):
        
        
        csv_logger = keras.callbacks.CSVLogger(runs_path/f'run_{RUN}'/'metrics'/f'fold_{fold}.log', separator=',', append=False)
        checkpoint = keras.callbacks.ModelCheckpoint(runs_path/f'run_{RUN}'/'models'/f'fold_{fold}.keras')
        tensorboard = keras.callbacks.TensorBoard(runs_path/f'run_{RUN}'/'tensorboard'/f'fold_{fold}')
        
        train_set = [SUBJECTS[i] for i in train_index]
        val_set = [SUBJECTS[i] for i in test_index]
        
        with open(runs_path/f'run_{RUN}'/f'fold_{fold}_hparams.json', "w") as f:
            json.dump(hparams | {'train_set':train_set, 'val_set':val_set}, f, indent=4)
        
        # train_dataset = make_dataset(train_set, output_signature, with_labels=True, batch_size=BATCH_SIZE)
        # val_dataset = make_dataset(val_set, output_signature, with_labels=True, batch_size=BATCH_SIZE)
        X_train, Y_train = make_X_Y(train_set, sisfall_dataset_processed_path)
        X_val, Y_val = make_X_Y(val_set, sisfall_dataset_processed_path)
        
        model = make_TinyCNN()
        output_bias_init(model)
        
        # adapt_normalization_layer(model, train_set, output_signature, BATCH_SIZE)
        adapt_normalization_layer(model, X_train)
        compile_model(model)
        
        model.fit(x=X_train, y=Y_train, shuffle=True, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val), callbacks=[csv_logger, checkpoint, tensorboard])
        
        break