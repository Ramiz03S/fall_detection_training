import numpy as np
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt

sisfall_dataset_processed_path = "PC\\datasets\\SisFall_dataset_processed\\"

subject_adult_list = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
subject_elderly_list = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]
subject_list = subject_adult_list + subject_elderly_list

output_signature_window = tf.TensorSpec(shape=(6,50,1), dtype=tf.float32)
output_signature_label = tf.TensorSpec(shape=(2,), dtype=tf.int32)
output_signature = (output_signature_window, output_signature_label)


def yield_subject_windows(subject, with_labels):
    subject = subject.decode('utf-8')
    with np.load(sisfall_dataset_processed_path + subject + ".npz") as subject_data:
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
                  shuffle_buffer_size=1000, cache_path=None):
    
    output_signature = output_signature if with_labels else output_signature[0]
    
    subject_ids_dataset = tf.data.Dataset.from_tensor_slices(subject_set)

    dataset = subject_ids_dataset.interleave(
        lambda subject: make_subject_dataset(subject, output_signature, with_labels),
        cycle_length= tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    dataset = dataset.cache(cache_path) if cache_path else dataset.cache()

    dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)

    if batch_size:
        dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def count_labels(subject_set):
    count = Counter()
    for subject in subject_set:
        with np.load(sisfall_dataset_processed_path + subject + ".npz") as subject_data:
            labels_oneshot_encoded = subject_data['labels'] # shape (n, 2)
            labels_unencoded = np.argmax(labels_oneshot_encoded, axis=1)
            count.update(labels_unencoded)
    return count

def count_labels_per_subject(subject_set):
    adl_array = []
    fall_array = []
    for subject in subject_set:
        with np.load(sisfall_dataset_processed_path + subject + ".npz") as subject_data:
            labels_oneshot_encoded = subject_data['labels']
            labels_unencoded = np.argmax(labels_oneshot_encoded, axis=1)
        count = Counter(labels_unencoded)
        adl_array.append(count[0])
        fall_array.append(count[1])
        
    return {"ADL": np.array(adl_array), "FALL": np.array(fall_array)}
        
def plot_subject_labels(subject_set):
    label_counts = count_labels_per_subject(subject_set)
    width = 0.3

    fig, ax = plt.subplots()
    bottom = np.zeros(len(subject_set))

    for boolean, label_count in label_counts.items():
        p = ax.bar(subject_set, label_count, width, label=boolean, bottom=bottom)
        bottom += label_count

    ax.legend(loc="upper right")

    plt.show()



'''

kf = KFold(n_splits=5, random_state=0, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(subject_adult_list)):
    print(f"Fold {i}: ")
    train_count = count_labels([subject_adult_list[idx] for idx in train_index])
    test_count = count_labels([subject_adult_list[idx] for idx in test_index])
    print(f"  Train Fall to ADL ratio: {train_count[2]/train_count[0]:.4f}")
    print(f"  Test Fall to ADL ratio:  {test_count[2]/test_count[0]:.4f}")
    #print(f"  Train: ", [subject_adult_list[idx] for idx in train_index])
    #print(f"  Test:  ", [subject_adult_list[idx] for idx in test_index])
    
'''

if __name__ == "__main__":
    pass
        
        
        
        
    
