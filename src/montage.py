import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Input, Output, State, ctx
import dash_daq as daq
from dotenv import load_dotenv
from numpy.lib.stride_tricks import sliding_window_view
from dataset_utils import normalize
import keras
import scipy

load_dotenv(override=True)

THRESHOLD = 0.54

colorscale = [
    [0.0, "#ffffff"],          
    [THRESHOLD, "#fcbba1"],
    [THRESHOLD, "#cb181d"],
    [1.0, "#67000d"],
]


DATASETS = {
    "sisfall": {"root": Path(os.environ["SISFALL_DATASET_LOADED_ROOT"]), "fs": 100},
    "kfall": {"root": Path(os.environ["KFALL_DATASET_LOADED_ROOT"]), "fs": 100},
}

AXES = ["x", "y", "z"]
COLORS = ["#d62728", "#2ca02c", "#1f77b4"]


def list_subjects(dataset):
    root = DATASETS[dataset]["root"]
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def list_trials(dataset, subject):
    if dataset == "sisfall":
        return [str(p) for p in sorted((DATASETS[dataset]["root"] / subject).glob("*.txt"))]
    else:
        return [str(p) for p in sorted((DATASETS[dataset]["root"] / subject).glob("*.csv"))]


def predict_sisfall(file):

    file_path = Path(file)
            
    trial_df = pd.read_csv(file)
    
    x = trial_df.iloc[:, :6].to_numpy(dtype=float)
    y = trial_df.iloc[:, 6].to_numpy()

    x_100 = scipy.signal.decimate(x, 2, axis=0)
    y_100 = pd.Series(y).groupby(np.arange(len(y)) // 2).max().to_numpy()

    trial_np = np.concatenate((x_100, y_100.reshape(-1, 1)), axis=1)
    
    # extending the fall phase by 0.5s at 100 Hz (50 samples)
    if file_path.name.startswith("F"):

        labels = trial_np[:, -1]
        fall_indicies = np.where(labels == 1)
        last_fall_index = fall_indicies[0][-1]
        extended_fall_range_end = min(labels.shape[0], last_fall_index + 50 + 1)
        labels[last_fall_index + 1: extended_fall_range_end] = 1
    
    # perform sliding windows of length 50 samples, and stride 10 samples
    windows_view = sliding_window_view(trial_np, 50, axis=0)[::10]


    X = windows_view[:,:-1].astype(np.float32).reshape((-1, 6, 50 , 1)) # shape (n, 6, 50 , 1)

    best_model_path = Path(os.environ.get("BEST_MODEL_PATH"))
    norm_stats_path = Path(os.environ.get("BEST_MODEL_NORM_STATS_PATH"))
    
    norm_stats = np.load(norm_stats_path)
    mean = norm_stats['mean'] 
    std = norm_stats['std'] 

    X = normalize(X, mean, std)

    model = keras.models.load_model(best_model_path, compile=False)

    fall_predictions = model.predict(X, batch_size = 512, verbose=0)[:, 1]

    return fall_predictions
    

def predict_kfall(file):

    file_path = Path(file)
            
    trial_df = pd.read_csv(file)
    
    x = trial_df.iloc[:, 1:7].to_numpy(dtype=float)
    y = trial_df.iloc[:, 7].to_numpy()

    
    trial_np = np.concatenate((x, y.reshape(-1, 1)), axis=1)
    
    # extending the fall phase by 0.5s at 100 Hz (50 samples)
    fall_frames = np.flatnonzero(y)
    if fall_frames.size:

        labels = trial_np[:, -1]
        fall_indicies = np.where(labels == 1)
        last_fall_index = fall_indicies[0][-1]
        extended_fall_range_end = min(labels.shape[0], last_fall_index + 50 + 1)
        labels[last_fall_index + 1: extended_fall_range_end] = 1
    
    # perform sliding windows of length 50 samples, and stride 10 samples
    windows_view = sliding_window_view(trial_np, 50, axis=0)[::10]


    X = windows_view[:,:-1].astype(np.float32).reshape((-1, 6, 50 , 1)) # shape (n, 6, 50 , 1)

    best_model_path = Path(os.environ.get("BEST_MODEL_PATH"))
    norm_stats_path = Path(os.environ.get("BEST_MODEL_NORM_STATS_PATH"))
    
    norm_stats = np.load(norm_stats_path)
    mean = norm_stats['mean'] 
    std = norm_stats['std'] 

    X = normalize(X, mean, std)

    model = keras.models.load_model(best_model_path, compile=False)

    fall_predictions = model.predict(X, batch_size = 512, verbose=0)[:, 1]

    return fall_predictions
  


def make_fig(file, dataset, fs, title, do_preds = True):

    trial_df = pd.read_csv(file)

    if dataset == "kfall":
        trial_acc = trial_df.iloc[:, 1:4].to_numpy(dtype=float)
        trial_gyro = trial_df.iloc[:, 4:7].to_numpy(dtype=float)
        trial_labels = trial_df.iloc[:, 7].to_numpy(dtype=int)
    elif dataset == "sisfall":

        x = trial_df.iloc[:, :6].to_numpy(dtype=float)
        y = trial_df.iloc[:, 6].to_numpy()

        x_100 = scipy.signal.decimate(x, 2, axis=0)
        y_100 = pd.Series(y).groupby(np.arange(len(y)) // 2).max().to_numpy()

        trial_acc = x_100[:, 0:3]
        trial_gyro = x_100[:, 3:6]
        trial_labels = y_100

    acc_norm = np.linalg.norm(trial_acc, axis=1)


    n = len(trial_acc)
    t = np.arange(n) / fs

    fall_probs = None

    if do_preds:
        if dataset == "sisfall":
            fall_probs = predict_sisfall(file)
        elif dataset == "kfall":
            fall_probs = predict_kfall(file)

    window, stride = 50, 10
    n_windows = (n - window) // stride + 1 
    t_prob = (np.arange(n_windows) * stride + window - 1) / fs   # end of each window


    if do_preds:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            row_heights=[0.45, 0.45, 0.1],
            subplot_titles=("Accelerometer", "Gyroscope", "Predicted fall probability"),
        )
    else:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            subplot_titles=("Accelerometer", "Gyroscope"),
        )

    for i, axis in enumerate(AXES):
        fig.add_trace(
            go.Scatter(x=t, y=trial_acc[:, i], name=axis, legendgroup=axis,
                       line=dict(color=COLORS[i], width=1)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=t, y=trial_gyro[:, i], name=axis, legendgroup=axis,
                       showlegend=False, line=dict(color=COLORS[i], width=1)),
            row=2, col=1,
        )

    fig.add_trace(
        go.Scatter(x=t, y=acc_norm, name="|acc|",
                   line=dict(color="black", width=1, dash="dot")),
        row=1, col=1,
    )
    if do_preds:
        fig.add_trace(
            go.Heatmap(
                x=t_prob, y=["P(fall)"], z=[fall_probs],
                colorscale=colorscale,
                zmin=0, zmax=1,
                colorbar=dict(title="P(fall)", thickness=10, len=0.25, y=0, yanchor="bottom"),
                hovertemplate="window end %{x:.2f} s<br>P(fall) %{z:.2f}<extra></extra>",
            ),
            row=3, col=1,
        )

    fall_frames = np.flatnonzero(trial_labels)
    if fall_frames.size:
        fig.add_vrect(
            x0=fall_frames[0] / fs, x1=fall_frames[-1] / fs,
            fillcolor="LightSalmon", opacity=0.3, layer="below", line_width=0,
        )
        fig.add_vrect(
            x0=fall_frames[-1] / fs, x1=(fall_frames[-1] + 50)/ fs,
            fillcolor="LightSalmon", opacity=0.15, layer="below", line_width=0,
        )

    fig.update_yaxes(title_text="acc (g)", row=1, col=1)
    fig.update_yaxes(title_text="gyro (°/s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)",
                     rangeslider=dict(visible=True, thickness=0.02),
                     row=3 if do_preds else 2, col=1)
    fig.update_yaxes(showticklabels=False, row=3, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1)
    fig.update_layout(
        title=title, height=700, template="plotly_white",
        hovermode="x unified", legend_title_text="axis",
    )
    return fig


app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Dropdown(id="dataset", options=list(DATASETS), value="kfall",
                     clearable=False, style={"width": "140px"}),
        dcc.Dropdown(id="subject", clearable=False, style={"width": "140px"}),
        html.Button("◀ prev", id="prev"),
        dcc.Dropdown(id="trial", clearable=False, style={"width": "320px"}),
        html.Button("next ▶", id="next"),
        daq.BooleanSwitch(id='predict', on=False, label="Prediction", labelPosition="buttom"),
    ], style={"display": "flex", "gap": "8px", "alignItems": "center"}),
    dcc.Graph(id="graph"),
])


# on change of dataset choice we update the list of subjects
@callback(
    Output("subject", "options"), Output("subject", "value"),
    Input("dataset", "value"),
)
def update_subjects(dataset):
    subjects = list_subjects(dataset)
    return subjects, (subjects[0] if subjects else None)


# on change of dataset or subject we update the list of trials
@callback(
    Output("trial", "options"), Output("trial", "value"),
    Input("dataset", "value"), Input("subject", "value"),
    Input("prev", "n_clicks"), Input("next", "n_clicks"),
    State("trial", "value"),
)
def update_trials(dataset, subject, _prev, _next, current):
    if not subject:
        return [], None
    trials = list_trials(dataset, subject)
    options = [{"label": Path(p).name, "value": p} for p in trials]
    if not trials:
        return options, None

    if ctx.triggered_id in ("prev", "next") and current in trials:
        step = 1 if ctx.triggered_id == "next" else -1
        return options, trials[(trials.index(current) + step) % len(trials)]
    return options, trials[0]


# on change of trial or predict button we update the figure
@callback(
    Output("graph", "figure"),
    Input("trial", "value"),
    Input("predict", "on"),
    State("dataset", "value"),
)
def update_graph(trial, predict, dataset):
    if not trial:
        return go.Figure()
    return make_fig(trial, dataset, DATASETS[dataset]["fs"], f"{dataset} dataset, trial file: {Path(trial).name}", do_preds=predict)


if __name__ == "__main__":
    app.run(debug=True)