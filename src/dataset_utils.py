import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import keras



CH_NAMES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

subject_adult_list = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
subject_elderly_list = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]

subject_list = subject_adult_list + subject_elderly_list
# 0 subjects are complete with falls and adl, 1 subjects just have adls
subject_type = [0 for _ in range(1,24)] + [1 for _ in range(1,6)] + [0] + [1 for _ in range(7,16)]

output_signature_window = tf.TensorSpec(shape=(6,50,1), dtype=tf.float32)
output_signature_label = tf.TensorSpec(shape=(2,), dtype=tf.int32)
output_signature = (output_signature_window, output_signature_label)


def yield_subject_windows(subject, with_labels):
    load_dotenv()  
    sisfall_dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))
    subject = subject.decode('utf-8')
    
    with np.load(sisfall_dataset_processed_path/f"{subject}.npz") as subject_data:
        windows = subject_data['windows']
        labels = subject_data['labels']
    for window, label in zip(windows, labels):
        yield (window, label) if with_labels else window

def make_subject_dataset(subject, output_signature, with_labels):    
    dataset = tf.data.Dataset.from_generator(
        yield_subject_windows,
        output_signature=output_signature,
        args=(subject, with_labels)
    )
    return dataset

def make_dataset(subject_set, output_signature, with_labels, batch_size=None,
                  shuffle_buffer_size=1000):
    
    output_signature = output_signature if with_labels else output_signature[0]
    
    subject_ids_dataset = tf.data.Dataset.from_tensor_slices(subject_set)

    dataset = subject_ids_dataset.interleave(
        lambda subject: make_subject_dataset(subject, output_signature, with_labels),
        cycle_length= tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    dataset = dataset.cache()

    dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size) if batch_size else dataset

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset



def count_labels(subject_set, sisfall_dataset_processed_path):
    count = Counter()
    for subject in subject_set:
        with np.load(sisfall_dataset_processed_path/f"{subject}.npz") as subject_data:
            labels_oneshot_encoded = subject_data['labels'] # shape (n, 2)
            labels_unencoded = np.argmax(labels_oneshot_encoded, axis=1)
            count.update(labels_unencoded)
    return count

def count_labels_stratified_kfold(n_splits, random_state, sisfall_dataset_processed_path):
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for fold, (train_index, test_index) in enumerate(skf.split(subject_list, subject_type)):
        
        train_set = [subject_list[i] for i in train_index]
        val_set = [subject_list[i] for i in test_index]
        train_count = count_labels(train_set, sisfall_dataset_processed_path)
        val_count = count_labels(val_set, sisfall_dataset_processed_path)
        
        print(f"fold {fold} train set has: {train_count[0]} ADLs, {train_count[1]} falls, {(train_count[1]/train_count[0]):.2f} fall to ADL ratio")
        print(f"fold {fold} val set has: {val_count[0]} ADLs, {val_count[1]} falls, {(val_count[1]/val_count[0]):.2f} fall to ADL ratio")
        

def count_labels_per_subject(subject_set, sisfall_dataset_processed_path):
    adl_array = []
    fall_array = []
    for subject in subject_set:
        with np.load(sisfall_dataset_processed_path/f"{subject}.npz") as subject_data:
            labels_oneshot_encoded = subject_data['labels']
            labels_unencoded = np.argmax(labels_oneshot_encoded, axis=1)
        count = Counter(labels_unencoded)
        adl_array.append(count[0])
        fall_array.append(count[1])
        
    return {"ADL": np.array(adl_array), "FALL": np.array(fall_array)}
        
def plot_subject_labels(subject_set, sisfall_dataset_processed_path):
    label_counts = count_labels_per_subject(subject_set, sisfall_dataset_processed_path)
    width = 0.3

    fig, ax = plt.subplots()
    bottom = np.zeros(len(subject_set))

    for boolean, label_count in label_counts.items():
        p = ax.bar(subject_set, label_count, width, label=boolean, bottom=bottom)
        bottom += label_count

    ax.legend(loc="upper right")

    plt.show()

 
def window_mean_std(X):
    X = np.squeeze(np.asarray(X), axis=-1)
    return X.mean(axis=2), X.std(axis=2)
 
def plot_before_after(windows, labels, norm_layer, title=""):
    
    is_fall = labels.argmax(axis=1) == 1
    windows_norm = np.asarray(norm_layer(windows))
 
    rows = {
        "before normalization": window_mean_std(windows),
        "globally normalized": window_mean_std(windows_norm),
    }
 
    fig, axes = plt.subplots(2, 6, figsize=(20, 7), sharex="row", sharey="row",
                             layout="constrained")
 
    for r, (row_name, (mu, sd)) in enumerate(rows.items()): 
        for c, ch in enumerate(CH_NAMES):
            ax = axes[r, c]
            
            for fall, name, color, size, alpha in [(False, "ADL", "#7f7f7f", 4, 0.3),
                                                   (True, "Fall", "#d62728", 8, 0.6)]:
                m = (is_fall == fall) & (sd[:, c] > 0)
                ax.scatter(mu[m, c], sd[m, c], s=size, color=color, alpha=alpha,
                           edgecolors="none", rasterized=True,
                           label=f"{name} (n={m.sum():,})")
            ax.set_yscale("log")
            ax.set_xlabel("window mean")
            if r == 0:
                ax.set_title(ch)
        axes[r, 0].set_ylabel(f"{row_name}\nwindow std (log)")
 
    leg = axes[0, 0].legend(markerscale=3, fontsize=8, loc="upper left")
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.suptitle(title)
    return fig


if __name__ == "__main__":
    
    load_dotenv()  
    sisfall_dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))

    layer = keras.layers.Normalization(axis=1, mean=[-0.01455288, -0.66143745, -0.08338201, -0.7908044,   2.058332,   -0.21747877], variance = [1.9349502e-01, 3.6206144e-01, 2.4309219e-01, 1.0970309e+03, 9.0329828e+02, 5.6539557e+02])
    # layer.adapt(make_dataset(subject_adult_list, output_signature=output_signature, batch_size=512, with_labels=False))
    # [-0.01455288 -0.66143745 -0.08338201 -0.7908044   2.058332   -0.21747877]
    # [1.9349502e-01 3.6206144e-01 2.4309219e-01 1.0970309e+03 9.0329828e+02 5.6539557e+02]
    
    with np.load(sisfall_dataset_processed_path/f"{subject_adult_list[1]}.npz") as subject_data:
        windows = subject_data['windows']
        labels = subject_data['labels']
        
    #fig = plot_before_after(windows, labels, layer,title="SA02: per-window mean vs std, before and after global normalization (layer adapted on fold 1 training subjects)")
    #fig.savefig("meanstd_before_after.png", dpi=300, bbox_inches="tight")
    #plt.show()
    
    #count_labels_stratified_kfold(5, 5, sisfall_dataset_processed_path)
    '''
    fold 0 train set has: 596892 ADLs, 14128 falls, 0.02 fall to ADL ratio
    fold 0 val set has: 158519 ADLs, 4144 falls, 0.03 fall to ADL ratio
    fold 1 train set has: 597120 ADLs, 14625 falls, 0.02 fall to ADL ratio
    fold 1 val set has: 158291 ADLs, 3647 falls, 0.02 fall to ADL ratio
    fold 2 train set has: 593633 ADLs, 14554 falls, 0.02 fall to ADL ratio
    fold 2 val set has: 161778 ADLs, 3718 falls, 0.02 fall to ADL ratio
    fold 3 train set has: 608645 ADLs, 14593 falls, 0.02 fall to ADL ratio
    fold 3 val set has: 146766 ADLs, 3679 falls, 0.03 fall to ADL ratio
    fold 4 train set has: 625354 ADLs, 15188 falls, 0.02 fall to ADL ratio
    fold 4 val set has: 130057 ADLs, 3084 falls, 0.02 fall to ADL ratio
    '''
    
    pass
        
        
        
        
    
