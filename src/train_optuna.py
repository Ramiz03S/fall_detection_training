import os
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from numpy import log, concatenate
import tensorflow as tf
import keras
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from dataset_utils import make_X_Y, fit_normalizer, normalize
from datetime import datetime
import optuna

STUDY_NAME = "study_3_kfall"
DATASET_USED = "kfall" # or "sisfall"
RAND_STATE = 5
SEARCH_SPACE = {"alpha":[0.5, 0.8, 0.85, 0.9], "gamma":[0, 2, 3, 4]}


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

if DATASET_USED == "sisfall":
    SUBJECTS = sisfall_subject_list
elif DATASET_USED == "kfall":
    SUBJECTS = kfall_subject_list
else:
    raise ValueError(f"Unknown DATASET_USED: {DATASET_USED}")

EPOCHS = 200
BATCH_SIZE = 512
RATE_FALLS = 0.10
LEARNING_RATE = 0.0005
N_SPLITS = 5
DATETIME = str(datetime.now().replace(microsecond=0))
TEST_SIZE = 0.15
FOLDS_RAN = 5
PATIENCE = 20
DELTA = 0.001
START_FROM_EPOCH = 20


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

def output_bias_init(tinyCNN_model_instance, rate_falls):
    dense_layer = tinyCNN_model_instance.layers[-1]
    dense_layer.bias.assign(log([1 - rate_falls, rate_falls]).astype("float32"))

def compile_model(tinyCNN_model_instance, threshold, alpha_falls, gamma, lr):
    tinyCNN_model_instance.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr), 
        jit_compile=False,
        loss=keras.losses.CategoricalFocalCrossentropy(alpha=[1 - alpha_falls, alpha_falls], gamma=gamma),
        metrics=[
            keras.metrics.SpecificityAtSensitivity(
                sensitivity=0.98, class_id=1, name="spec_at_sens98", num_thresholds=1000
            ),
            # Recall divides true_positives by the sum of true_positives and false_negatives
            keras.metrics.Recall(class_id=1, name="sensitivity", thresholds=threshold), 
            # TP (correct falls) / (TP (correct falls) + FN (Falls predicted as ADLs))
            # Of the actual falls, how many did the model catch (as falls)?
            # miss rate is (1 − sensitivity)
            keras.metrics.Recall(class_id=0, name="specificity", thresholds=1-threshold), 
            # TP (correct ADLs) / (TP (correct ADLs) + FN (ADLs predicted as Falls))
            # Of the actual negatives (ADL), how many did the model correctly reject (as falls)? 
            # 1 − specificity is the false alarm rate
            
        ]
    )
    
def threshold_sweep(fall_probs, labels):
    results_dict = {}
    fpr, tpr, thresholds = roc_curve(labels, fall_probs, drop_intermediate=False)

    table = pd.DataFrame({
        "threshold":   thresholds,
        "sensitivity": tpr,
        "specificity": 1 - fpr,
    })

    achieved_sensitivity = table[table["sensitivity"] >= 0.98]
    best_specificity = achieved_sensitivity.sort_values(["specificity", "sensitivity"], ascending=False).iloc[0]            
            
    results_dict["sensitivity"] = best_specificity["sensitivity"]
    results_dict["specificity"] = best_specificity["specificity"]
    results_dict["threshold"] = best_specificity["threshold"]
            
    return results_dict     

def train(trial : optuna.Trial):
    
    ALPHA_FALLS = trial.suggest_categorical("alpha", choices=SEARCH_SPACE["alpha"])
    GAMMA = trial.suggest_categorical("gamma", choices=SEARCH_SPACE["gamma"])
    
    RUN = f"optuna_run_a_{ALPHA_FALLS}_g_{GAMMA}"
    RUN_DESCRIPTION = ""
    
    
    devices = tf.config.list_physical_devices('GPU')
    print("number of gpu devices: ", len(devices)) 
    print("printing tf.test.is_built_with_cuda(): ", tf.test.is_built_with_cuda())
    
    load_dotenv(override=True)

    if DATASET_USED == "sisfall":
        dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))
    elif DATASET_USED == "kfall":
        dataset_processed_path = Path(os.environ.get("KFALL_DATASET_PROCESSED_ROOT"))

    print("Data path:", dataset_processed_path.resolve())
    optuna_runs_path = Path(os.environ.get("OPTUNA_RUNS_ROOT"))/DATASET_USED
   
    (optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'models').mkdir(parents=True, exist_ok=True)
    (optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'metrics').mkdir(parents=True, exist_ok=True)
    (optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'tensorboard').mkdir(parents=True, exist_ok=True)
    
    if DATASET_USED == "sisfall":
        train_val_set, test_set, train_val_type, test_set_type = train_test_split(sisfall_subject_list, sisfall_subject_type_list, test_size=TEST_SIZE, random_state=RAND_STATE, shuffle=True, stratify=sisfall_subject_type_list)
        skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RAND_STATE, shuffle=True)
        train_val_gen = skf.split(train_val_set, train_val_type)
    elif DATASET_USED == "kfall":
        train_val_set, test_set = train_test_split(kfall_subject_list, test_size=TEST_SIZE, random_state=RAND_STATE, shuffle=True)
        kf = KFold(n_splits=N_SPLITS, random_state=RAND_STATE, shuffle=True)
        train_val_gen = kf.split(train_val_set)
    
    # X_test, Y_test = make_X_Y(test_set, dataset_processed_path)
    
    hparams = {
        "date_time": DATETIME,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "alpha": ALPHA_FALLS,
        "gamma": GAMMA,
        "rate_falls": RATE_FALLS,
        "n_splits": N_SPLITS,
        "rand_state": RAND_STATE,
        "patience" : PATIENCE,
        "delta" : DELTA,
        "start_from_epoch" :START_FROM_EPOCH,
        "run_description": RUN_DESCRIPTION,
        "subjects_used": SUBJECTS,
        "dataset used" : DATASET_USED
    }
    
    with open(optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/f'test_params.json', "w") as f:
        json.dump({'test_set':test_set, 'test_size':TEST_SIZE}, f, indent=4)
    
    folds_fall_probs = []
    folds_labels = []
    
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
        
        csv_logger = keras.callbacks.CSVLogger(optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'metrics'/f'fold_{fold}.log', 
                                               separator=',',
                                               append=False)
        checkpoint = keras.callbacks.ModelCheckpoint(optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'models'/f'fold_{fold}.keras', 
                                                     monitor="val_spec_at_sens98", save_best_only=True, mode="max")
        tensorboard = keras.callbacks.TensorBoard(optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'tensorboard'/f'fold_{fold}')
        earlystopping = keras.callbacks.EarlyStopping(monitor="val_spec_at_sens98", 
                                                      mode="max", patience=PATIENCE, 
                                                      start_from_epoch=START_FROM_EPOCH, 
                                                      min_delta=DELTA)
        
        with open(optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/f'fold_{fold}_hparams.json', "w") as f:
            json.dump(hparams | {'train_set':train_set, 'val_set':val_set}, f, indent=4)
                
        
        model = make_TinyCNN()
        output_bias_init(model, rate_falls=RATE_FALLS)

        #adapt_normalization_layer(model, X_train)
        mean, std = fit_normalizer(X_train, 
                                   save_path=optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/f'norm_stats_fold_{fold}.npz')

        compile_model(model, threshold=0.5, alpha_falls=ALPHA_FALLS, gamma=GAMMA, lr=LEARNING_RATE)
        model.fit(
            x=normalize(X_train, mean, std),
            y=Y_train,
            shuffle=True, batch_size=BATCH_SIZE, epochs=EPOCHS,
            validation_data=(normalize(X_val, mean, std), Y_val),
            callbacks=[csv_logger, checkpoint, tensorboard, earlystopping],
            verbose=2)
        
        
        best_model = keras.models.load_model(optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'models'/f'fold_{fold}.keras',
                                             compile=False)
        Y_val_predict = best_model.predict(normalize(X_val, mean, std), batch_size=BATCH_SIZE)
        fall_probs = Y_val_predict[:,1]
        labels = Y_val.argmax(axis=1) 
        
        folds_fall_probs.append(fall_probs)
        folds_labels.append(labels)
    
    all_fall_probs = concatenate(folds_fall_probs)
    all_labels = concatenate(folds_labels)
    
    results = threshold_sweep(all_fall_probs, all_labels)
    
    trial.set_user_attr("sensitivity", float(results["sensitivity"]))
    trial.set_user_attr("specificity", float(results["specificity"]))
    trial.set_user_attr("threshold", float(results["threshold"]))
    
    return float(results["specificity"])

def optuna_init(study_name):
    load_dotenv(override=True)  
    optuna_runs_path = Path(os.environ.get("OPTUNA_RUNS_ROOT"))
    (optuna_runs_path).mkdir(parents=True, exist_ok=True)
    
    sampler = optuna.samplers.GridSampler(search_space=SEARCH_SPACE, seed=RAND_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, storage=f"sqlite:///{str(optuna_runs_path/'tuning.db')}", study_name=study_name, load_if_exists=True)

    # guard against appending trials from one dataset to a study made on another
    study_dataset = study.user_attrs.get("dataset")
    if study_dataset is None:
        study.set_user_attr("dataset", DATASET_USED)
    elif study_dataset != DATASET_USED:
        raise ValueError(f"Study '{study_name}' was created for '{study_dataset}', but DATASET_USED is '{DATASET_USED}'. Use a different STUDY_NAME.")

    return study       

if __name__ == "__main__":
    
    study = optuna_init(STUDY_NAME)
    
    study.optimize(train, n_trials=16)