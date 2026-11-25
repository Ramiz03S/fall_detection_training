import os
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from dataset_utils import make_X_Y, fit_normalizer, normalize
from datetime import datetime

RUN = "99925_a_0.99_G_2.4_t_0.549_wN_all_maxspec_decimate"
RUN_DESCRIPTION = "Preporcessing done using decimate. Following optuna trial 4, with alpha=0.99, gamma=2.4, threshold=0.549, and all subjects used for creating sets, normalization is done from outside the model, and normalization stats are saved for later use in inference. The model is trained for 200 epochs with a batch size of 512, and the best model is saved based on validation specificity."
DATASET_USED = "kfall" # or "sisfall"
EPOCHS = 1
BATCH_SIZE = 512
ALPHA_FALLS = 0.99
GAMMA = 2.4
THRESHOLD_FALLS = 0.549
RATE_FALLS = 0.03
LEARNING_RATE = 0.0005
N_SPLITS = 5
RAND_STATE = 5
DATETIME = str(datetime.now().replace(microsecond=0))
TEST_SIZE = 0.15
FOLDS_RAN = 5
SUBJECTS = None


sisfall_subject_adult_list = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
sisfall_subject_elderly_list = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]

sisfall_subject_list = sisfall_subject_adult_list + sisfall_subject_elderly_list
# 0 subjects are complete with falls and adl, 1 subjects just have adls
sisfall_subject_type_list = [0 for _ in range(1,24)] + [1 for _ in range(1,6)] + [0] + [1 for _ in range(7,16)]

kfall_subject_list = [
    "SA06", "SA07", "SA08", "SA09", "SA10", "SA11", "SA12", "SA13",
    "SA14", "SA15", "SA16", "SA17", "SA18", "SA19", "SA20", "SA21",
    "SA22", "SA23", "SA24", "SA25", "SA26", "SA27", "SA28", "SA29",
    "SA30", "SA31", "SA32", "SA33", "SA35", "SA36", "SA37", "SA38",
]

def make_TinyCNN():
    
    inputs = keras.Input(shape=(6, 50, 1))
    # normalization = keras.layers.Normalization(axis=1)(inputs)
    temp_conv = keras.layers.Conv2D(filters=8, kernel_size=(1,16), padding="same")(inputs)
    relu1 = keras.layers.Activation('relu')(temp_conv)
    spatial_conv = keras.layers.Conv2D(filters=8, kernel_size=(6,1), padding="same")(relu1)
    relu2 = keras.layers.Activation('relu')(spatial_conv)
    global_avg_pooling = keras.layers.GlobalAveragePooling2D()(relu2)
    outputs = keras.layers.Dense(units=2, activation="softmax")(global_avg_pooling)
    
    return keras.Model(inputs=inputs, outputs=outputs)

# deprecated
def adapt_normalization_layer(tinyCNN_model_instance, X_train):
    layer = tinyCNN_model_instance.get_layer(index=1)
    layer.adapt(X_train)

def output_bias_init(tinyCNN_model_instance):
    dense_layer = tinyCNN_model_instance.layers[-1]
    dense_layer.bias.assign(np.log([1 - RATE_FALLS, RATE_FALLS]).astype("float32"))

def compile_model(tinyCNN_model_instance, lr = LEARNING_RATE, alpha_falls = ALPHA_FALLS, gamma = GAMMA):
    tinyCNN_model_instance.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr), 
        jit_compile=False,
        loss=keras.losses.CategoricalFocalCrossentropy(alpha=[1 - alpha_falls, alpha_falls], gamma=gamma),
        metrics=[
            # Recall divides true_positives by the sum of true_positives and false_negatives
            keras.metrics.Recall(class_id=1, name="sensitivity", thresholds=THRESHOLD_FALLS), 
            # TP (correct falls) / (TP (correct falls) + FN (Falls predicted as ADLs))
            # Of the actual falls, how many did the model catch (as falls)?
            # miss rate is (1 − sensitivity)
            keras.metrics.Recall(class_id=0, name="specificity", thresholds=1-THRESHOLD_FALLS), 
            # TP (correct ADLs) / (TP (correct ADLs) + FN (ADLs predicted as Falls))
            # Of the actual negatives (ADL), how many did the model correctly reject (as falls)? 
            # 1 − specificity is the false alarm rate
        ])

def main():
    
    devices = tf.config.list_physical_devices('GPU')
    print("number of gpu devices: ", len(devices)) 
    print("printing tf.test.is_built_with_cuda(): ", tf.test.is_built_with_cuda())
    
    load_dotenv(override=True)

    if DATASET_USED == "sisfall":
        dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))
        SUBJECTS = sisfall_subject_list
    elif DATASET_USED == "kfall":
        dataset_processed_path = Path(os.environ.get("KFALL_DATASET_PROCESSED_ROOT"))
        SUBJECTS = kfall_subject_list

    print("Data path:", dataset_processed_path.resolve())
    runs_path = (Path(os.environ.get("RUNS_ROOT")))/DATASET_USED
   
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
        "subjects_used": SUBJECTS,
        "dataset used" : DATASET_USED         
    }

    if DATASET_USED == "sisfall":
        train_val_set, test_set, train_val_type, test_set_type = train_test_split(sisfall_subject_list, sisfall_subject_type_list, test_size=TEST_SIZE, random_state=RAND_STATE, shuffle=True, stratify=sisfall_subject_type_list)
        skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RAND_STATE, shuffle=True)
        train_val_gen = skf.split(train_val_set, train_val_type)
    elif DATASET_USED == "kfall":
        train_val_set, test_set = train_test_split(kfall_subject_list, test_size=TEST_SIZE, random_state=RAND_STATE, shuffle=True)
        kf = KFold(n_splits=N_SPLITS, random_state=RAND_STATE, shuffle=True)
        train_val_gen = kf.split(train_val_set)

    # X_test, Y_test = make_X_Y(test_set, sisfall_dataset_processed_path)
    
    with open(runs_path/f'run_{RUN}'/f'test_params.json', "w") as f:
        json.dump({'test_set':test_set, 'test_size':TEST_SIZE}, f, indent=4)

    for fold, (train_index, val_index) in enumerate(train_val_gen):
        
        keras.backend.clear_session()
        keras.utils.set_random_seed(RAND_STATE)

        if fold == FOLDS_RAN:
            break
        
        train_set = [train_val_set[i] for i in train_index]
        val_set = [train_val_set[i] for i in val_index]
        
        X_train, Y_train = make_X_Y(train_set, dataset_processed_path)
        X_val, Y_val = make_X_Y(val_set, dataset_processed_path)
        assert np.isfinite(X_train).all() and np.isfinite(X_val).all(), "Dataset contains NaN or Inf"
        
        csv_logger = keras.callbacks.CSVLogger(runs_path/f'run_{RUN}'/'metrics'/f'fold_{fold}.log', 
                                               separator=',', 
                                               append=False)
        checkpoint = keras.callbacks.ModelCheckpoint(runs_path/f'run_{RUN}'/'models'/f'fold_{fold}.keras',
                                                     monitor='val_specificity',
                                                     mode='max',
                                                     save_best_only=True, verbose=1)
        tensorboard = keras.callbacks.TensorBoard(runs_path/f'run_{RUN}'/'tensorboard'/f'fold_{fold}')
        
        with open(runs_path/f'run_{RUN}'/f'fold_{fold}_hparams.json', "w") as f:
            json.dump(hparams | {'train_set':train_set, 'val_set':val_set}, f, indent=4)

        model = make_TinyCNN()
        output_bias_init(model)

        mean, std = fit_normalizer(X_train,
                                   save_path=runs_path/f'run_{RUN}'/f'norm_stats_fold_{fold}.npz')

        compile_model(model)
        model.fit(
            x=normalize(X_train, mean, std), 
            y=Y_train,
            shuffle=True, batch_size=BATCH_SIZE, epochs=EPOCHS,
            validation_data=(normalize(X_val, mean, std), Y_val),
            callbacks=[csv_logger, checkpoint, tensorboard])

if __name__ == "__main__":

    main()