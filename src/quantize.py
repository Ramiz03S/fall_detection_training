import json

import numpy as np
import keras
import tensorflow as tf
from pathlib import Path
from dotenv import load_dotenv
import os
from ai_edge_litert.interpreter import Interpreter
from dataset_utils import make_X_Y

def normalize(X, mean, std):
    return ((X - mean) / std)

def calib_labels(Y_train):
    global calib
    Y_train = Y_train.argmax(1)  
    rng = np.random.default_rng(0)                                   
    falls = np.flatnonzero(Y_train == 1)                             
    adls  = np.flatnonzero(Y_train == 0)
    n = min(250, len(falls))                                         
    calib = np.concatenate([rng.choice(falls, n, replace=False),
                            rng.choice(adls, n, replace=False)])     

def representative_dataset():
    for i in calib:
        yield [X_train[i:i+1].astype(np.float32)]   

def quantize_model(model_path, export_path):

    model = keras.models.load_model(model_path, compile=False)
    model.export(export_path/"best_model", include_optimizer=False)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(export_path/"best_model"))
    f32_bytes = converter.convert()
    tf.lite.experimental.Analyzer.analyze(model_content=f32_bytes)
    # model without quantization, just converted into tflite format
    (export_path/"tflite_format").mkdir(parents=True, exist_ok=True)
    open(export_path/"tflite_format"/"f32.tflite", "wb").write(f32_bytes) 


    converter = tf.lite.TFLiteConverter.from_saved_model(str(export_path/"best_model"))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]                        
    converter.representative_dataset = representative_dataset                   
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] 
    converter.inference_input_type = tf.int8                                    
    converter.inference_output_type = tf.int8                                   
    tflite_int8 = converter.convert()               
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_int8)
    open(export_path/"tflite_format"/"int8.tflite", "wb").write(tflite_int8)


def main():
    global X_train
    
    load_dotenv()  
    sisfall_dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))
    model_path = Path(os.environ.get("BEST_MODEL_PATH"))
    hparams_path = Path(os.environ.get("BEST_MODEL_HPARAMS_PATH"))
    norm_stats_path = Path(os.environ.get("BEST_MODEL_NORM_STATS_PATH"))

    export_path = Path(os.environ.get("QUANTIZE_EXPORT_PATH"))
    export_path.mkdir(parents=True, exist_ok=True)

    with open(hparams_path, 'r') as hparams_json:
        hparams = json.load(hparams_json)

    train_set = hparams["train_set"]
    X_train, Y_train = make_X_Y(train_set, sisfall_dataset_processed_path)

    norm_stats = np.load(norm_stats_path)
    mean = norm_stats['mean'] # (1, 6, 1, 1)
    std = norm_stats['std'] # (1, 6, 1, 1)

    X_train = normalize(X_train, mean, std)


    calib_labels(Y_train)

    # print(X_train[calib].min(), X_train[calib].max())      
    # print(X_train.min(), X_train.max())      

    quantize_model(model_path, export_path)

    
    it = Interpreter(model_path=export_path/"tflite_format"/"int8.tflite")               
    it.allocate_tensors()                                            
    inp = it.get_input_details()[0]                                  
    out = it.get_output_details()[0]
    print("input :", inp["shape"], inp["dtype"], inp["quantization"])  
    print("output:", out["shape"], out["dtype"], out["quantization"])


if __name__ == "__main__":
    main()