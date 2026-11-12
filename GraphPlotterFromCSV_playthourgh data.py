from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_iteration_envelopes(
    file_paths: list[str],
    colors: list[str] | None = None,
    xlabel: str = "Iteration",
    ylabel: str = "Metric Value",
    window_size: int = 5,
    envelope_smooth_window: int = 5,
    band_alpha: float = 0.25,
    line_width: float = 2.5,
    title: str = "Iteration Performance Comparison",
    save_path: str | None = None,
):
  """Plots iteration-based data after sorting by the iteration (X-axis) column.

  Calculates row-wise mean (bold curve) and min/max (envelope band) across
  replicate columns, followed by smoothing along the ordered iterations.

  Parameters:
  ----------
  file_paths : list of str
      Paths to CSV files. Column 0 is the iteration (X-axis); remaining
      columns are data points/replicates for that iteration.
  colors : list of str, optional
      Line/fill colors for each file. Defaults to Matplotlib cycle if None.
  xlabel : str, default='Iteration'
      Label for the X-axis.
  ylabel : str, default='Metric Value'
      Label for the Y-axis.
  window_size : int, default=5
      Rolling window size for the bold mean curve. Set to 1 for no smoothing.
  envelope_smooth_window : int, default=5
      Rolling window size for the min/max band edges. Set to 1 for no smoothing.
  band_alpha : float, default=0.25
      Transparency (0 to 1) of the shaded envelope.
  line_width : float, default=2.5
      Stroke width for the bold mean trendline.
  title : str, default='Iteration Performance Comparison'
      Graph title.
  save_path : str, optional
      Path to save the generated image file.
  """
  n_files = len(file_paths)

  # Color cycle fallback
  if colors is None:
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]
    colors = [default_colors[i % len(default_colors)] for i in range(n_files)]
  elif len(colors) < n_files:
    raise ValueError(
        f"Provided {len(colors)} colors, but need at least {n_files}."
    )

  fig, ax = plt.subplots(figsize=(10, 6))

  for path_str, color in zip(file_paths, colors):
    path = Path(path_str)
    series_name = path.stem

    df = pd.read_csv(path)

    # 1. Sort ascending by the first column (Iteration) and reset index
    iter_col = df.columns[0]
    df = df.sort_values(by=iter_col).reset_index(drop=True)

    # First column is the sorted X-axis
    x = df.iloc[:, 0]

    # Remaining columns are the data points for each iteration
    data_points = df.iloc[:, 1:]

    # 2. Compute per-row statistics across replicate columns
    row_mean = data_points.mean(axis=1)
    row_min = data_points.min(axis=1)
    row_max = data_points.max(axis=1)

    # 3. Apply rolling smoothing across the ordered iteration axis
    if window_size > 1:
      mean_smoothed = row_mean.rolling(
          window=window_size, center=True, min_periods=1
      ).mean()
    else:
      mean_smoothed = row_mean

    if envelope_smooth_window > 1:
      min_smoothed = row_min.rolling(
          window=envelope_smooth_window, center=True, min_periods=1
      ).mean()
      max_smoothed = row_max.rolling(
          window=envelope_smooth_window, center=True, min_periods=1
      ).mean()
    else:
      min_smoothed = row_min
      max_smoothed = row_max

    # 4. Plot shaded fluctuation envelope (min to max)
    ax.fill_between(
        x,
        min_smoothed,
        max_smoothed,
        color=color,
        alpha=band_alpha,
        linewidth=0,
        label="_nolegend_",
    )

    # 5. Plot bold average curve
    ax.plot(
        x,
        mean_smoothed,
        color=color,
        linewidth=line_width,
        label=series_name,
    )

  ax.set_xlabel(xlabel, fontsize=12)
  ax.set_ylabel(ylabel, fontsize=12)
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
    r"D:\PongRL2\Graphs\epsilonDecay_rewards\epsilon Decay end 100000.csv",
    r"D:\PongRL2\Graphs\epsilonDecay_rewards\epsilon Decay end 200000.csv",
    r"D:\PongRL2\Graphs\epsilonDecay_rewards\epsilon Decay end 300000.csv"
]



custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green


plot_iteration_envelopes(
    file_paths=files,
    colors=custom_colors,
    xlabel="Training Iterations",
    ylabel="Reward",
    window_size=7,  # Smoothing applied to the bold mean line
    envelope_smooth_window=7,  # Smoothing applied to the min/max band edges
    band_alpha=0.20,  # Opacity of the min/max band
    line_width=2.8,  # Bold graph stroke width
    title="Model Convergence & Fluctuation per Iteration",
  )