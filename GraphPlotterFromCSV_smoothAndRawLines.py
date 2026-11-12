from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd


def plot_combined_files(
    file_paths: list[str],
    x_cols: Union[list[str], str],
    y_cols: Union[list[str], str],
    colors: list[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    window_size: int = 7,
    show_raw: bool = False,
    title: str = "Combined Data Comparison",
    save_path: str | None = None,
):
  """Plots multiple datasets on a single Matplotlib graph with customizable

  axis labels, column selection, colors, and smoothing.

  Parameters:
  ----------
  file_paths : list of str
      Paths to the data files.
  x_cols : list of str or str
      Column name(s) for the X-axis (list per file or single string if shared).
  y_cols : list of str or str
      Column name(s) for the Y-axis (list per file or single string if shared).
  colors : list of str, optional
      List of colors for each series. Uses default cycle if None.
  xlabel : str, optional
      Custom label for the X-axis. If None, defaults to the CSV column name.
  ylabel : str, optional
      Custom label for the Y-axis. If None, defaults to the CSV column name.
  window_size : int, default=7
      Rolling average window size. Set to 1 for raw/unsmoothed data.
  show_raw : bool, default=False
      If True, plots faint raw data lines behind the smoothed curves.
  title : str, default='Combined Data Comparison'
      Title of the plot.
  save_path : str, optional
      If provided, saves the figure to this path.
  """
  n_files = len(file_paths)

  # Normalize column inputs to lists matching file count
  if isinstance(x_cols, str):
    x_cols = [x_cols] * n_files
  elif len(x_cols) != n_files:
    raise ValueError(
        f"Length of x_cols ({len(x_cols)}) must match file_paths ({n_files})."
    )

  if isinstance(y_cols, str):
    y_cols = [y_cols] * n_files
  elif len(y_cols) != n_files:
    raise ValueError(
        f"Length of y_cols ({len(y_cols)}) must match file_paths ({n_files})."
    )

  # Setup colors
  if colors is None:
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [default_colors[i % len(default_colors)] for i in range(n_files)]
  elif len(colors) < n_files:
    raise ValueError(
        f"Provided {len(colors)} colors, but need at least {n_files}."
    )

  fig, ax = plt.subplots(figsize=(10, 6))

  for path_str, x_col, y_col, color in zip(file_paths, x_cols, y_cols, colors):
    path = Path(path_str)
    series_name = path.stem  # Filename without extension

    df = pd.read_csv(path)
    x = df[x_col]
    y = df[y_col]

    # Centered rolling mean smoothing
    if window_size > 1:
      y_smoothed = y.rolling(
          window=window_size, center=True, min_periods=1
      ).mean()

      # alpha between 0 (very smooth) and 1 (raw data)
      y_smoothed = y.ewm(alpha=0.01, adjust=False).mean()

    else:
      y_smoothed = y

    # Raw noisy trace (optional)
    if show_raw and window_size > 1:
      ax.plot(x, y, color=color, alpha=0.25, linewidth=1, label="_nolegend_")

    # Smoothed main line
    ax.plot(
        x,
        y_smoothed,
        color=color,
        linewidth=2,
        label=f"{series_name} (smoothed)",
    )

  # Set axis labels: uses custom label if supplied, otherwise falls back to CSV column names
  final_xlabel = (
      xlabel
      if xlabel is not None
      else (x_cols[0] if len(set(x_cols)) == 1 else "X Axis")
  )
  final_ylabel = (
      ylabel
      if ylabel is not None
      else (y_cols[0] if len(set(y_cols)) == 1 else "Y Axis")
  )

  ax.set_xlabel(final_xlabel, fontsize=12)
  ax.set_ylabel(final_ylabel, fontsize=12)
  ax.set_title(title, fontsize=14, fontweight="bold")
  ax.grid(True, linestyle="--", alpha=0.5)
  ax.legend(frameon=True)
  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to {save_path}")

  plt.show()

#decay rate
files = [
    r"D:\PongRL2\Graphs\epsilonDecay\epsilon Decay end 100000.csv",
    r"D:\PongRL2\Graphs\epsilonDecay\epsilon Decay end 200000.csv",
    r"D:\PongRL2\Graphs\epsilonDecay\epsilon Decay end 300000.csv"
]

# Case 2: If all files share the exact same column names, you can pass strings:
x_columns = "Step"
y_columns = "Value"

line_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

plot_combined_files(
    file_paths=files,
    x_cols=x_columns,
    y_cols=y_columns,
    colors=line_colors,
    xlabel="iterations",  # Custom X-axis label
    ylabel="Output Voltage (mV)",  # Custom Y-axis label
    window_size=15,  # Control smoothing level
    show_raw=True,  # Set True to display the raw noisy data behind the lines
    title="Epsilon Decay rates",
)