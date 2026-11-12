from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd


def plot_smoothed_envelope(
    file_paths: list[str],
    x_cols: Union[list[str], str],
    y_cols: Union[list[str], str],
    colors: list[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    window_size: int = 15,
    envelope_smooth_window: int = 9,
    band_alpha: float = 0.25,
    title: str = "Smoothed Signal with Fluctuation Envelope",
    save_path: str | None = None,
):
  """Plots smoothed trendlines with an independently smoothed min/max envelope band.

  Parameters:
  ----------
  file_paths : list of str
      Paths to the data files.
  x_cols : list of str or str
      Column name(s) for the X-axis (list per file or single string if shared).
  y_cols : list of str or str
      Column name(s) for the Y-axis (list per file or single string if shared).
  colors : list of str, optional
      List of colors for each series. Defaults to Matplotlib cycle if None.
  xlabel : str, optional
      Custom label for the X-axis. Defaults to CSV column header if None.
  ylabel : str, optional
      Custom label for the Y-axis. Defaults to CSV column header if None.
  window_size : int, default=15
      Rolling window size used to calculate the mean trend and initial min/max.
  envelope_smooth_window : int, default=9
      Rolling window size applied to the min and max series to eliminate
      blocky/staircase artifacts. Set to 1 for un-smoothed min/max edges.
  band_alpha : float, default=0.25
      Transparency (0 to 1) of the shaded envelope band.
  title : str, default='Smoothed Signal with Fluctuation Envelope'
      Title of the plot.
  save_path : str, optional
      Path to save the generated figure.
  """
  n_files = len(file_paths)

  # Normalize column inputs
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
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]
    colors = [default_colors[i % len(default_colors)] for i in range(n_files)]
  elif len(colors) < n_files:
    raise ValueError(
        f"Provided {len(colors)} colors, but need at least {n_files}."
    )

  fig, ax = plt.subplots(figsize=(10, 6))

  for path_str, x_col, y_col, color in zip(file_paths, x_cols, y_cols, colors):
    path = Path(path_str)
    series_name = path.stem

    df = pd.read_csv(path)
    x = df[x_col]
    y = df[y_col]

    # 1. Base rolling calculation (mean and raw local extremes)
    roller = y.rolling(window=window_size, center=True, min_periods=1)
    y_smoothed = roller.mean()
    y_min = roller.min()
    y_max = roller.max()

    # 2. Smooth the min and max boundaries to eliminate staircase transitions
    if envelope_smooth_window > 1:
      y_min = y_min.rolling(
          window=envelope_smooth_window, center=True, min_periods=1
      ).mean()
      y_max = y_max.rolling(
          window=envelope_smooth_window, center=True, min_periods=1
      ).mean()

    # Shaded smoothed ribbon
    ax.fill_between(
        x,
        y_min,
        y_max,
        color=color,
        alpha=band_alpha,
        linewidth=0,
        label="_nolegend_",
    )

    # Primary smoothed line
    ax.plot(
        x,
        y_smoothed,
        color=color,
        linewidth=2.2,
        label=f"{series_name} (smoothed)",
    )

  # Resolve axis labels
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

plot_smoothed_envelope(
    file_paths=files,
    x_cols=x_columns,
    y_cols=y_columns,
    colors=line_colors,
    xlabel="iteration",
    ylabel="Voltage (V)",
    window_size=50,  # Window size for the mean and the min/max band
    envelope_smooth_window=20,  # Smoothing applied to min/max band edges
    band_alpha=0.25,  # Band opacity (0.1 = very light, 0.5 = dark)
    title="Win ratio average, over 10 episodes ",
  )