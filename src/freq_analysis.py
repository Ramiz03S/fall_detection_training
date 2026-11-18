import os
from pathlib import Path
import random
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

acc_columns = ["acc_X", "acc_Y", "acc_Z"]
gyro_columns = ["gyro_X", "gyro_Y", "gyro_Z"]

SubjectAdultList = [f"SA{str(i).zfill(2)}" for i in range(1, 24)]
SubjectElderlyList = [f"SE{str(i).zfill(2)}" for i in range(1, 16)]
subject_list = SubjectAdultList + SubjectElderlyList


def downsample_trial(trial_df, method):
    """Returns (trial_np, fs, nperseg) for 'original', 'decimate', 'average'."""
    trial_np = trial_df.to_numpy()

    if method == "original":
        return trial_np, 200, 512
    elif method == "decimate":
        return scipy.signal.decimate(trial_np[:, :6].astype(float), 2, axis=0), 100, 256
    elif method == "average":
        df = trial_df.copy()
        df.index = pd.to_timedelta(np.arange(len(df)) / 200, unit="s")
        return df.iloc[:, :6].resample("10ms").mean().to_numpy(), 100, 256
    raise ValueError(f"method must be 'original', 'decimate', or 'average', got {method!r}")

def welch_compare(sisfall_dataset_loaded_path, subject_list=subject_list,
                  methods=("original", "decimate", "average")):
    """One figure per trial: every method's PSD overlaid on the same axes."""
    colors = {"original": "black", "decimate": "tab:blue", "average": "tab:orange"}
    channels = ["acc_X", "acc_Y", "acc_Z", "gyro_X", "gyro_Y", "gyro_Z"]
    plt.rcParams["font.size"] = 15

    for subject in subject_list:

        folder = f"Subject_{subject}_frequency_analysis_welch_compare"
        plot_save_path = Path(os.environ.get("PLOT_SAVE_ROOT"))
        (plot_save_path/folder).mkdir(parents=True, exist_ok=True)

        for file in (sisfall_dataset_loaded_path/subject).iterdir():

            trial_df = pd.read_csv(file)

            fig, axs = plt.subplots(3, 2, figsize=(24, 14), sharex=True)
            fig.suptitle(f'Welch PSD of {file.name[:-4]}: downsampling methods compared', fontsize=20)
            lowest = np.full(6, np.inf)   # lowest point of the original curve, per channel

            for method in methods:
                trial_np, fs, nperseg = downsample_trial(trial_df, method)
                f, psd = scipy.signal.welch(trial_np[:, :6], fs=fs, axis=0, return_onesided=True, nperseg=nperseg, scaling='density', detrend='linear')
                psd_db = 10*np.log10(psd)
                if method == "original":
                    lowest = psd_db.min(axis=0)

                for c in range(6):
                    axs[c % 3, c // 3].plot(f, psd_db[:, c], label=method, color=colors[method], linewidth=1)

            for c, ax in enumerate(axs.T.flat):   # acc X/Y/Z down the left, gyro X/Y/Z down the right
                ax.set_title(channels[c], fontsize=17, fontweight='bold')
                if np.isfinite(lowest[c]):
                    ax.set_ylim(bottom=lowest[c] - 10)   # stop decimate's -120 dB drop from squashing the plot
                ax.axvline(50, color='grey', linestyle=':', linewidth=1)   # new Nyquist
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('PSD (dB)')
                ax.tick_params(labelbottom=True)
                ax.margins(x=0)   # no empty space left/right of the curves

            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', ncol=len(methods), fontsize=16)   # one legend for the figure
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            plt.savefig(plot_save_path/folder/f"{file.name[:-4]}.png", dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    load_dotenv()  
    sisfall_dataset_loaded_path = Path(os.environ.get("SISFALL_DATASET_LOADED_ROOT"))

    welch_compare(sisfall_dataset_loaded_path, subject_list=[subject_list[0]], methods=("original", "decimate", "average"))
