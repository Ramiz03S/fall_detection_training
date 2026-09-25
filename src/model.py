import os
import json
from pathlib import Path
from dotenv import load_dotenv
import tensorflow as tf
import keras
from sklearn.model_selection import KFold
from dataset_utils import make_dataset

RUN = 0
EPOCHS = 3
BATCH_SIZE = 512
ALPHA = 0.25
GAMMA = 2
LEARNING_RATE = 0.0005
N_SPLITS = 5
RAND_STATE = 5

output_signature_window = tf.TensorSpec(shape=(6,50,1), dtype=tf.float32)
output_signature_label = tf.TensorSpec(shape=(2,), dtype=tf.int32)
output_signature = (output_signature_window, output_signature_label)

subject_adult_list = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
subject_elderly_list = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]
subject_list = subject_adult_list + subject_elderly_list

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

def adapt_normalization_layer(tinyCNN_model_instance, train_set, output_signature, batch_size):
    layer = tinyCNN_model_instance.get_layer(index=1)
    layer.adapt(make_dataset(train_set, output_signature=output_signature, batch_size=batch_size, with_labels=False))

def compile_model(tinyCNN_model_instance, lr = LEARNING_RATE, alpha = ALPHA, gamma = GAMMA):
    tinyCNN_model_instance.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr), 
        jit_compile=False,
        loss=keras.losses.CategoricalFocalCrossentropy(alpha=[alpha, 1 - alpha], gamma=gamma),
        metrics=[
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.Precision(class_id=1),
            keras.metrics.Recall(class_id=1),
        ])


if __name__ == "__main__":
    
    
    devices = tf.config.list_physical_devices('GPU')
    print("number of gpu devices: ", len(devices)) 
    print("printing tf.test.is_built_with_cuda(): ", tf.test.is_built_with_cuda())
    
    load_dotenv()  
    
    sisfall_dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))
    
    model_checkpoint_path = Path(os.environ.get("MODEL_CHECKPOINT_ROOT"))
    model_logs_path = Path(os.environ.get("MODEL_LOGS_ROOT"))
    tensorboard_logs_path = Path(os.environ.get("TENSORBOARD_LOGS_ROOT"))
    
    (model_checkpoint_path/f'run_{RUN}').mkdir(parents=True, exist_ok=True)
    (model_logs_path/f'run_{RUN}').mkdir(parents=True, exist_ok=True)
    (tensorboard_logs_path/f'run_{RUN}').mkdir(parents=True, exist_ok=True)
    
    hparams = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "n_splits": N_SPLITS,
        "rand_state": RAND_STATE,
        "dataset_used": SUBJECTS            
    }
    with open(model_logs_path/f'run_{RUN}'/"hparams.json", "w") as f:
        json.dump(hparams, f, indent=4)
    
    kf = KFold(n_splits=N_SPLITS, random_state=RAND_STATE, shuffle=True)

    for fold, (train_index, test_index) in enumerate(kf.split(SUBJECTS)):
        
        
        csv_logger = keras.callbacks.CSVLogger(model_logs_path/f'run_{RUN}'/f'fold_{fold}.log', separator=',', append=False)
        checkpoint = keras.callbacks.ModelCheckpoint(model_checkpoint_path/f'run_{RUN}'/f'fold_{fold}.keras', save_best_only=True)
        tensorboard = tf.keras.callbacks.TensorBoard(tensorboard_logs_path/f'run_{RUN}'/'logs')
        
        train_set = [SUBJECTS[i] for i in train_index]
        val_set = [SUBJECTS[i] for i in test_index]
        
        train_dataset = make_dataset(train_set, output_signature, with_labels=True, batch_size=BATCH_SIZE)
        val_dataset = make_dataset(val_set, output_signature, with_labels=True, batch_size=BATCH_SIZE)
        
        model = make_TinyCNN()
        
        adapt_normalization_layer(model, train_set, output_signature, BATCH_SIZE)
        compile_model(model)
        
        model.fit(x=train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=val_dataset, callbacks=[csv_logger, checkpoint, tensorboard])
        
        break
        
    