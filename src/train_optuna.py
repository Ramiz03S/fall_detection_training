import os
import json
from pathlib import Path
from dotenv import load_dotenv
from numpy import log, concatenate
import tensorflow as tf
import keras
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from dataset_utils import make_X_Y
from datetime import datetime
import optuna

STUDY_NAME = "testing_optuna1"
RAND_STATE = 5
SEARCH_SPACE = {"alpha":[0.25,0.5,0.75], "gamma":[1, 1.5, 2]}
THRESHOLD_SPACE = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

subject_adult_list = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
subject_elderly_list = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]

subject_list = subject_adult_list + subject_elderly_list
subject_type_list = [0 for _ in range(1,24)] + [1 for _ in range(1,6)] + [0] + [1 for _ in range(7,16)]

EPOCHS = 1
BATCH_SIZE = 512
RATE_FALLS = 0.03
LEARNING_RATE = 0.0005
N_SPLITS = 5
DATETIME = str(datetime.now().replace(microsecond=0))
TEST_SIZE = 0.15
FOLDS_RAN = 1
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

    thresholds = THRESHOLD_SPACE
    results_dict = {"max_mean_metrics": -1}
    
    for threshold in thresholds:
        preds = (fall_probs > threshold).astype(int)
        
        sensitivity = recall_score(y_true=labels, y_pred=preds, pos_label=1)
        specificity = recall_score(y_true=labels, y_pred=preds, pos_label=0)
        
        mean_metrics = (sensitivity + specificity) / 2
        if mean_metrics > results_dict["max_mean_metrics"]:
            results_dict["max_mean_metrics"] = mean_metrics 
            results_dict["sensitivity"] = sensitivity
            results_dict["specificity"] = specificity
            results_dict["threshold"] = threshold
            
    return results_dict     

def train(trial : optuna.Trial):
    
    ALPHA_FALLS = trial.suggest_categorical("alpha", choices=SEARCH_SPACE["alpha"])
    GAMMA = trial.suggest_categorical("gamma", choices=SEARCH_SPACE["gamma"])
    
    RUN = f"optuna_run_a_{ALPHA_FALLS}_g_{GAMMA}"
    RUN_DESCRIPTION = ""
    
    
    devices = tf.config.list_physical_devices('GPU')
    print("number of gpu devices: ", len(devices)) 
    print("printing tf.test.is_built_with_cuda(): ", tf.test.is_built_with_cuda())
    
    load_dotenv()  
    sisfall_dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))
    optuna_runs_path = Path(os.environ.get("OPTUNA_RUNS_ROOT"))
   
    (optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'models').mkdir(parents=True, exist_ok=True)
    (optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'metrics').mkdir(parents=True, exist_ok=True)
    (optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'tensorboard').mkdir(parents=True, exist_ok=True)
    
    train_val_set, test_set, train_val_type, test_set_type = train_test_split(subject_list, subject_type_list, test_size=TEST_SIZE, random_state=RAND_STATE, shuffle=True, stratify=subject_type_list)
    
    # X_test, Y_test = make_X_Y(test_set, sisfall_dataset_processed_path)
    
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
        "run_description": RUN_DESCRIPTION,
        "dataset_used": SUBJECTS
    }
    
    with open(optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/f'test_params.json', "w") as f:
        json.dump({'test_set':test_set, 'test_size':TEST_SIZE}, f, indent=4)
    
    folds_fall_probs = []
    folds_labels = []
    
    skf = StratifiedKFold(n_splits=N_SPLITS, random_state=RAND_STATE, shuffle=True)
    for fold, (train_index, val_index) in enumerate(skf.split(train_val_set, train_val_type)):
        
        keras.backend.clear_session()
        
        if fold == FOLDS_RAN:
            break
        
        train_set = [train_val_set[i] for i in train_index]
        val_set = [train_val_set[i] for i in val_index]
        
        X_train, Y_train = make_X_Y(train_set, sisfall_dataset_processed_path)
        X_val, Y_val = make_X_Y(val_set, sisfall_dataset_processed_path)
        
        csv_logger = keras.callbacks.CSVLogger(optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'metrics'/f'fold_{fold}.log', separator=',', append=False)
        checkpoint = keras.callbacks.ModelCheckpoint(optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'models'/f'fold_{fold}.keras')
        tensorboard = keras.callbacks.TensorBoard(optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/'tensorboard'/f'fold_{fold}')
        
        with open(optuna_runs_path/f'study_{STUDY_NAME}'/f'run_{RUN}'/f'fold_{fold}_hparams.json', "w") as f:
            json.dump(hparams | {'train_set':train_set, 'val_set':val_set}, f, indent=4)
                
        
        model = make_TinyCNN()
        output_bias_init(model, rate_falls=RATE_FALLS)
        adapt_normalization_layer(model, X_train)
        compile_model(model, threshold=0.5, alpha_falls=ALPHA_FALLS, gamma=GAMMA, lr=LEARNING_RATE)
        model.fit(
            x=X_train, y=Y_train,
            shuffle=True, batch_size=BATCH_SIZE, epochs=EPOCHS,
            validation_data=(X_val, Y_val),
            callbacks=[csv_logger, checkpoint, tensorboard])
        
        Y_val_predict = model.predict(X_val, batch_size=BATCH_SIZE)
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
    
    return float(results["max_mean_metrics"])

def optuna_init(study_name):
    load_dotenv()  
    optuna_runs_path = Path(os.environ.get("OPTUNA_RUNS_ROOT"))
    (optuna_runs_path).mkdir(parents=True, exist_ok=True)
    
    sampler = optuna.samplers.GridSampler(search_space=SEARCH_SPACE, seed=RAND_STATE)
    return optuna.create_study(direction="maximize", sampler=sampler, storage=f"sqlite:///{str(optuna_runs_path/"tuning.db")}", study_name=study_name, load_if_exists=True)       

if __name__ == "__main__":
    
    study = optuna_init(STUDY_NAME)
    
    study.optimize(train, n_trials=9)