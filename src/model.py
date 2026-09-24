import os
from pathlib import Path
from dotenv import load_dotenv
import tensorflow as tf
import keras
from sklearn.model_selection import KFold
from dataset_utils import make_dataset

EPOCHS = 1
BATCH_SIZE = 512
ALPHA = 0.25
GAMMA = 2
LEARNING_RATE = 0.0005
N_SPLITS = 5

output_signature_window = tf.TensorSpec(shape=(6,50,1), dtype=tf.float32)
output_signature_label = tf.TensorSpec(shape=(2,), dtype=tf.int32)
output_signature = (output_signature_window, output_signature_label)

subject_adult_list = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
subject_elderly_list = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]
subject_list = subject_adult_list + subject_elderly_list

def make_TinyCNN():
    
    inputs = keras.Input(shape=(6, 50, 1))
    normalization = keras.layers.Normalization(axis=1)(inputs)
    temp_conv = keras.layers.Conv2D(filters=8, kernel_size=(1,16), padding="same", activation="relu")(normalization)
    spatial_conv = keras.layers.Conv2D(filters=8, kernel_size=(6,1), padding="same", activation="relu")(temp_conv)
    global_avg_pooling = keras.layers.GlobalAveragePooling2D()(spatial_conv)
    outputs = keras.layers.Dense(units=2, activation="softmax")(global_avg_pooling)
    
    return keras.Model(inputs=inputs, outputs=outputs)

def adapt_normalization_layer(tinyCNN_model_instance, train_set, output_signature, batch_size):
    layer = tinyCNN_model_instance.get_layer(index=1)
    layer.adapt(make_dataset(train_set, output_signature=output_signature, batch_size=batch_size, with_labels=False))

def compile_model(tinyCNN_model_instance, lr = LEARNING_RATE, alpha = ALPHA, gamma = GAMMA):
    tinyCNN_model_instance.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr), 
        loss=keras.losses.CategoricalFocalCrossentropy(alpha=alpha, gamma=gamma),
        metrics=[
            keras.metrics.Accuracy(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.AUC(),
        ])


if __name__ == "__main__":
    
    load_dotenv()  
    sisfall_dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))
    
    kf = KFold(n_splits=N_SPLITS, random_state=5, shuffle=True)

    for fold, (train_index, test_index) in enumerate(kf.split(subject_adult_list)):
        
        csv_logger = keras.callbacks.CSVLogger(f'training_fold_{fold}.log', separator=',', append=False)
        train_set = [subject_adult_list[i] for i in train_index]
        val_set = [subject_adult_list[i] for i in test_index]
        
        train_dataset = make_dataset(train_set, output_signature, with_labels=True, batch_size=BATCH_SIZE)
        val_dataset = make_dataset(val_set, output_signature, with_labels=True, batch_size=BATCH_SIZE)
        
        model = make_TinyCNN()
        
        adapt_normalization_layer(model, train_set, output_signature, BATCH_SIZE)
        compile_model(model)
        
        model.fit(x=train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=val_dataset, callbacks=[csv_logger])
        
    