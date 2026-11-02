import json
import numpy as np
import keras
import tensorflow as tf
from pathlib import Path
from dotenv import load_dotenv
import os
from ai_edge_litert.interpreter import Interpreter
from dataset_utils import make_X_Y
from sklearn.metrics import confusion_matrix
import pandas as pd


def normalize(X, mean, std):
    return ((X - mean) / std)

def calib_labels(Y_train):
    global calib
    rng = np.random.default_rng(0)                                   
    falls = np.flatnonzero(Y_train == 1)                             
    adls  = np.flatnonzero(Y_train == 0)
    n = min(250, len(falls))                                         
    calib = np.concatenate([rng.choice(falls, n, replace=False),
                            rng.choice(adls, n, replace=False)])     

def representative_dataset():
    for i in calib:
        yield [X_train[i:i+1].astype(np.float32)]   

def quantize_model(best_model_path, export_path):

    model = keras.models.load_model(best_model_path, compile=False)
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

def keras_fall_prob(best_model_path, X):

    model = keras.models.load_model(best_model_path, compile=False)
    return model.predict(X, batch_size=512)[:, 1]

def tflite_fall_prob(path, X):
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    print("input :", inp["shape"], inp["dtype"], inp["quantization"])  
    print("output:", out["shape"], out["dtype"], out["quantization"])
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]

    p = np.empty(len(X), dtype=np.float32)                   
    for k in range(len(X)):
        x = X[k:k+1]                                         
        if inp["dtype"] == np.int8:                          
            x = np.clip(np.round(x / s_in) + z_in, -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], x)
        it.invoke()
        y = it.get_tensor(out["index"])[0]                   
        if out["dtype"] == np.int8:                          
            y = (y.astype(np.float32) - z_out) * s_out
        p[k] = y[1]                                          
    return p

def report(y, p, thr=0.5):
    pred = (p >= thr).astype(int)                                       
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()  
    return {
        "sens": tp / (tp + fn) if tp + fn else float("nan"),            
        "spec": tn / (tn + fp) if tn + fp else float("nan"),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,                       
    }

def main():
    global X_train
    
    load_dotenv()  
    sisfall_dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))
    best_model_path = Path(os.environ.get("BEST_MODEL_PATH"))
    hparams_path = Path(os.environ.get("BEST_MODEL_HPARAMS_PATH"))
    norm_stats_path = Path(os.environ.get("BEST_MODEL_NORM_STATS_PATH"))

    export_path = Path(os.environ.get("QUANTIZE_EXPORT_PATH"))
    export_path.mkdir(parents=True, exist_ok=True)

    with open(hparams_path, 'r') as hparams_json:
        hparams = json.load(hparams_json)

    train_set = hparams["train_set"]
    val_set = hparams["val_set"]
    X_train, Y_train = make_X_Y(train_set, sisfall_dataset_processed_path)
    X_val, Y_val = make_X_Y(val_set, sisfall_dataset_processed_path)

    norm_stats = np.load(norm_stats_path)
    mean = norm_stats['mean'] # (1, 6, 1, 1)
    std = norm_stats['std'] # (1, 6, 1, 1)

    X_train = normalize(X_train, mean, std)
    Y_train = Y_train.argmax(1)
    X_val = normalize(X_val, mean, std)
    Y_val = Y_val.argmax(1)

    calib_labels(Y_train)

    # print(X_train[calib].min(), X_train[calib].max())      
    # print(X_train.min(), X_train.max())      

    quantize_model(best_model_path, export_path)

    data = {
        "train": (X_train, Y_train),
        "val":   (X_val, Y_val),
    } 

    results = {}
    for name, (X, y) in data.items():                                   
        results[name] = {
            "y":     y,
            "keras": keras_fall_prob(best_model_path, X),
            "f32":   tflite_fall_prob(export_path/"tflite_format"/"f32.tflite", X),
            "int8":  tflite_fall_prob(export_path/"tflite_format"/"int8.tflite", X),
        }                                                                 

    np.savez(export_path/"eval_probs.npz",
            **{f"{dataset}_{mode}": values for dataset, result in results.items() for mode, values in result.items()}) 
    # For each train/val dataset, save the true y, and fall probabilities for keras, f32 and int8 models 
    # in the npz file under keys like "train_y", "train_keras", "train_f32", "train_int8", "val_y", "val_keras", "val_f32", "val_int8"
    

    rows = []
    for dataset, result in results.items():
        for mode in ["keras", "f32", "int8"]:
            rows.append({"set": dataset, "model": mode, **report(result["y"], result[mode])})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))

if __name__ == "__main__":
    main()