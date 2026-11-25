import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import random
import sys

CH_NAMES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

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

# output_signature_window = tf.TensorSpec(shape=(6,50,1), dtype=tf.float32)
# output_signature_label = tf.TensorSpec(shape=(2,), dtype=tf.int32)
# output_signature = (output_signature_window, output_signature_label)

# deprecated
def yield_subject_windows(subject, with_labels):
    load_dotenv()  
    sisfall_dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))
    subject = subject.decode('utf-8')
    
    with np.load(sisfall_dataset_processed_path/f"{subject}.npz") as subject_data:
        windows = subject_data['windows']
        labels = subject_data['labels']
    for window, label in zip(windows, labels):
        yield (window, label) if with_labels else window

# deprecated
def make_subject_dataset(subject, output_signature, with_labels):    
    dataset = tf.data.Dataset.from_generator(
        yield_subject_windows,
        output_signature=output_signature,
        args=(subject, with_labels)
    )
    return dataset
# deprecated
def make_dataset(subject_set, output_signature, with_labels, batch_size=None,
                  shuffle_buffer_size=1000):
    random.seed(0)
    random.shuffle(subject_set)
    
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

# deprecated
def plot_frac_from_generator(subject_list, output_signature, batch_size, shuffle_buffer_size, save_dir):
    
    dataset = make_dataset(subject_list, output_signature, True, batch_size, shuffle_buffer_size)
    fracs = [y.numpy()[:, 1].mean() for _, y in dataset]
        
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(fracs)), fracs, linewidth=0.8)

    ax.set(xlabel='batch', ylabel='fraction of falls in batch',
        title= 'Fraction of Falls in Batches of Shuffled Dataset from Generator with shuffle buffer of size {shuffle_buffer_size}')
    ax.grid()

    fig.savefig(save_dir/f"Fall_fractions_g_{shuffle_buffer_size}.png")


def make_X_Y(subject_set, dataset_processed_path):
    
    with np.load(dataset_processed_path/f"{subject_set[0]}.npz") as subject_data:
                X, Y = (subject_data['windows'], subject_data['labels'])
    for subject in subject_set[1:]:
        with np.load(dataset_processed_path/f"{subject}.npz") as subject_data:
            X = np.concatenate([X, subject_data['windows']], axis=0)
            Y = np.concatenate([Y, subject_data['labels']], axis=0)
    return X, Y  

def plot_frac_from_Y(Y, batch_size, rand_state, save_dir, shuffle=True, filename = ''):
    
    if shuffle:
        rng = np.random.default_rng(seed=rand_state)
        rng.shuffle(Y)
        
    num_samples = Y.shape[0]
    num_full_batches = num_samples // batch_size
    remainder = num_samples % batch_size
    
    batches_idx = [i for i in range(num_full_batches)]
    
    
    fracs = [np.mean(Y[idx * batch_size: (idx + 1) * batch_size, 1]) for idx in batches_idx]
    fracs.append(np.mean(Y[num_full_batches * 512: num_full_batches * 512 + remainder, 1]))
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(num_full_batches + 1), fracs, linewidth=0.8)

    ax.set(xlabel='batch', ylabel='fraction of falls in batch',
        title= 'Fraction of Falls in Batches of Shuffled Dataset'if shuffle else 'Fraction of Falls in Batches of Unshuffled Dataset')
    ax.grid()

    fig.savefig(save_dir/f"Fall_fractions_{'shufled' if shuffle else 'unshufled'}_{filename}.png")

def count_labels(subject_set, dataset_processed_path):
    count = Counter()
    for subject in subject_set:
        with np.load(dataset_processed_path/f"{subject}.npz") as subject_data:
            labels_oneshot_encoded = subject_data['labels'] # shape (n, 2)
            labels_unencoded = np.argmax(labels_oneshot_encoded, axis=1)
            count.update(labels_unencoded)
    return count

# only on sisfall dataset
def count_labels_stratified_kfold(n_splits, subject_list, random_state, dataset_processed_path):
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for fold, (train_index, test_index) in enumerate(skf.split(subject_list, subject_type)):
        
        train_set = [subject_list[i] for i in train_index]
        val_set = [subject_list[i] for i in test_index]
        train_count = count_labels(train_set, dataset_processed_path)
        val_count = count_labels(val_set, dataset_processed_path)
        
        print(f"fold {fold} train set has: {train_count[0]} ADLs, {train_count[1]} falls, {(train_count[1]/train_count[0]):.2f} fall to ADL ratio")
        print(f"fold {fold} val set has: {val_count[0]} ADLs, {val_count[1]} falls, {(val_count[1]/val_count[0]):.2f} fall to ADL ratio")

# only on sisfall dataset
def count_labels_stratified_kfolds_wTestset(test_size, subject_list, n_splits, random_state, dataset_processed_path):
    
    subject_train, subject_test, subject_train_type, subject_test_type = train_test_split(subject_list, sisfall_subject_type_list, test_size=test_size, random_state=random_state, shuffle=True, stratify=sisfall_subject_type_list)
    
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    
    for fold, (train_index, val_index) in enumerate(skf.split(subject_train, subject_train_type)):
        
        train_set = [subject_train[i] for i in train_index]
        val_set = [subject_train[i] for i in val_index]
        
        train_count = count_labels(train_set, dataset_processed_path)
        val_count = count_labels(val_set, dataset_processed_path)
        test_count = count_labels(subject_test, dataset_processed_path)
        
        print(f"fold {fold} train set has: {train_count[0]} ADLs, {train_count[1]} falls, {(train_count[1]/train_count[0]):.3f} fall to ADL ratio")
        print(f"fold {fold} val set has: {val_count[0]} ADLs, {val_count[1]} falls, {(val_count[1]/val_count[0]):.3f} fall to ADL ratio")
    print(f"test set has: {test_count[0]} ADLs, {test_count[1]} falls, {(test_count[1]/test_count[0]):.3f} fall to ADL ratio")
        
# only on kfall dataset
def count_labels_kfolds_wTestset(test_size, subject_list, n_splits, random_state, dataset_processed_path):
    
    subject_train, subject_test = train_test_split(subject_list, test_size=test_size, random_state=random_state, shuffle=True)
    
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    
    for fold, (train_index, val_index) in enumerate(kf.split(subject_train)):
        
        train_set = [subject_train[i] for i in train_index]
        val_set = [subject_train[i] for i in val_index]
        
        train_count = count_labels(train_set, dataset_processed_path)
        val_count = count_labels(val_set, dataset_processed_path)
        test_count = count_labels(subject_test, dataset_processed_path)
        
        print(f"fold {fold} train set has: {train_count[0]} ADLs, {train_count[1]} falls, {(train_count[1]/train_count[0]):.3f} fall to ADL ratio")
        print(f"fold {fold} val set has: {val_count[0]} ADLs, {val_count[1]} falls, {(val_count[1]/val_count[0]):.3f} fall to ADL ratio")
    print(f"test set has: {test_count[0]} ADLs, {test_count[1]} falls, {(test_count[1]/test_count[0]):.3f} fall to ADL ratio")
       

def count_labels_per_subject(subject_set, dataset_processed_path):
    adl_array = []
    fall_array = []
    for subject in subject_set:
        with np.load(dataset_processed_path/f"{subject}.npz") as subject_data:
            labels_oneshot_encoded = subject_data['labels']
            labels_unencoded = np.argmax(labels_oneshot_encoded, axis=1)
        count = Counter(labels_unencoded)
        adl_array.append(count[0])
        fall_array.append(count[1])
        
    return {"ADL": np.array(adl_array), "FALL": np.array(fall_array)}
        
def plot_subject_labels(subject_set, dataset_processed_path, save_dir, filename = ''):
    label_counts = count_labels_per_subject(subject_set, dataset_processed_path)
    width = 0.3

    fig, ax = plt.subplots(layout="constrained")
    bottom = np.zeros(len(subject_set))

    for boolean, label_count in label_counts.items():
        p = ax.bar(subject_set, label_count, width, label=boolean, bottom=bottom)
        bottom += label_count

    fall_percentages = 100 * np.asarray(label_count) / bottom
    ax.bar_label(p, labels=[f"{percent:.1f}%" for percent in fall_percentages])

    ax.legend(loc="upper right")
    ax.grid()
    ax.set_axisbelow(True)
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(save_dir/f"{filename}_subjects_label_distribution.png", dpi=300)


def fit_normalizer(X_train, save_path=None):

    mean = X_train.mean(axis=(0, 2, 3), keepdims=True, dtype=np.float64)
    std = X_train.std(axis=(0, 2, 3), keepdims=True, dtype=np.float64)
    std = np.maximum(std, 1e-7)
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    if save_path is not None:
        np.savez(save_path, mean=mean, std=std)
    return mean, std

def normalize(X, mean, std):
    return ((X - mean) / std)

# need fixing
def window_mean_std(X):
    X = np.squeeze(np.asarray(X), axis=-1)
    return X.mean(axis=2), X.std(axis=2)
# need fixing
def plot_before_after_norm(subject, dataset_processed_path, norm_layer, title, save_dir):
    
    with np.load(dataset_processed_path/f"{subject}.npz") as subject_data:
        windows = subject_data['windows']
        labels = subject_data['labels']
    
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
    
    fig.savefig(save_dir/f"meanstd_before_after_{subject}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":

  
    old_stdout = sys.stdout
    log_file = open("dataset_utils.log","w", encoding='utf-8')
    sys.stdout = log_file

    load_dotenv()  
    sisfall_dataset_processed_path = Path(os.environ.get("SISFALL_DATASET_PROCESSED_ROOT"))
    kfall_dataset_processed_path = Path(os.environ.get("KFALL_DATASET_PROCESSED_ROOT"))

    plot_save_path = Path(os.environ.get("PLOT_SAVE_ROOT"))
    plot_save_path.mkdir(parents=True, exist_ok=True)

    X_kfall, Y_kfall = make_X_Y(kfall_subject_list, kfall_dataset_processed_path)
    X_sisfall, Y_sisfall = make_X_Y(sisfall_subject_list, sisfall_dataset_processed_path)
    
    plot_frac_from_Y(Y_sisfall, 512, 0, plot_save_path, filename='sisfall_all_subjects')
    plot_frac_from_Y(Y_kfall, 512, 0, plot_save_path, filename='kfall_all_subjects')

    count_sisfall = count_labels(sisfall_subject_list, sisfall_dataset_processed_path)
    count_kfall = count_labels(kfall_subject_list, kfall_dataset_processed_path)
    print(f"Ratio of falls to ADLs across all subjects in sisfall dataset: {np.round(count_sisfall[1] / count_sisfall[0], 3)}")
    print(f"Ratio of falls to ADLs across all subjects in kfall dataset: {np.round(count_kfall[1] / count_kfall[0], 3)}")


    mean_sisfall, std_sisfall = fit_normalizer(X_sisfall)
    mean_kfall, std_kfall = fit_normalizer(X_kfall)

    print("mean and std of sisfall dataset: ")
    for i, ch in enumerate(CH_NAMES):
        print(f"{ch} : {mean_sisfall.flatten()[i]:.3f} ± {std_sisfall.flatten()[i]:.3f}")
    print("mean and std of kfall dataset: ")
    for i, ch in enumerate(CH_NAMES):
        print(f"{ch} : {mean_kfall.flatten()[i]:.3f} ± {std_kfall.flatten()[i]:.3f}")

    plot_subject_labels(kfall_subject_list, kfall_dataset_processed_path, plot_save_path, filename='kfall_all_subjects_label_distribution')
    plot_subject_labels(sisfall_subject_list, sisfall_dataset_processed_path, plot_save_path, filename='sisfall_all_subjects_label_distribution')

    
    count_labels_stratified_kfolds_wTestset(test_size=0.15, subject_list=sisfall_subject_list, n_splits=5, random_state=1, dataset_processed_path=sisfall_dataset_processed_path)
    count_labels_kfolds_wTestset(test_size=0.15, subject_list=kfall_subject_list, n_splits=5, random_state=1, dataset_processed_path=kfall_dataset_processed_path)

    sys.stdout = old_stdout
    log_file.close()
   