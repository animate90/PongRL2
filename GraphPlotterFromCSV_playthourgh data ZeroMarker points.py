from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def darken_color(color: str, factor: float = 0.65) -> tuple[float, float, float]:
  """Darkens any Matplotlib color (hex or named) by scaling its RGB channels."""
  r, g, b = mcolors.to_rgb(color)
  return (max(0.0, r * factor), max(0.0, g * factor), max(0.0, b * factor))


def plot_iteration_scatter_envelopes(
    file_paths: list[str],
    colors: list[str] | None = None,
    xlabel: str = "Iteration",
    ylabel: str = "Metric Value",
    window_size: int = 5,
    spread_ratio: float = 0.80,
    spread_mode: str = "linear",
    point_size: float = 18.0,
    point_alpha: float = 0.35,
    below_zero_darken_factor: float = 0.60,
    below_zero_alpha: float = 0.45,
    line_width: float = 2.5,
    highlight_trough: bool = True,
    title: str = "Iteration Performance with Spread Data Points",
    save_path: str | None = None,
):
  """Plots iteration data replacing min/max shaded bands with individual data points

  spread along the X-axis between each iteration and the next.

  Parameters:
  ----------
  file_paths : list of str
      Paths to CSV files. Column 0 is the iteration index; remaining columns are
      replicates.
  colors : list of str, optional
      Base colors for each series.
  xlabel : str, default='Iteration'
      X-axis label.
  ylabel : str, default='Metric Value'
      Y-axis label.
  window_size : int, default=5
      Smoothing window for the bold mean curve. Set to 1 for raw averages.
  spread_ratio : float, default=0.80
      Fraction of the interval to the next iteration across which points are
      spread
      (e.g., 0.80 spans 80% of the distance to the next step, leaving a 20%
      buffer).
  spread_mode : {'linear', 'random'}, default='linear'
      'linear' places points at evenly spaced intervals between current and next
      x.
      'random' jitters points uniformly across the interval.
  point_size : float, default=18.0
      Marker size for the replicate scatter points.
  point_alpha : float, default=0.35
      Transparency (0 to 1) of the scatter points.
  below_zero_darken_factor : float, default=0.60
      Brightness multiplier for highlighting values < 0 (< 1.0 is darker).
  below_zero_alpha : float, default=0.45
      Fill opacity for shaded negative regions.
  line_width : float, default=2.5
      Line thickness of the bold mean trendline.
  highlight_trough : bool, default=True
      If True, annotates the lowest negative dip below zero.
  title : str, default='Iteration Performance with Spread Data Points'
      Graph title.
  save_path : str, optional
      Path to save the generated image file.
  """
  n_files = len(file_paths)

  if colors is None:
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]
    colors = [default_colors[i % len(default_colors)] for i in range(n_files)]
  elif len(colors) < n_files:
    raise ValueError(
        f"Provided {len(colors)} colors, but need at least {n_files}."
    )

  fig, ax = plt.subplots(figsize=(10, 6))

  # 1. Zero baseline
  ax.axhline(
      0,
      color="#444444",
      linestyle="--",
      linewidth=1.2,
      alpha=0.8,
      label="Baseline (0)",
      zorder=2,
  )

  for path_str, base_color in zip(file_paths, colors):
    path = Path(path_str)
    series_name = path.stem
    dark_tone = darken_color(base_color, factor=below_zero_darken_factor)

    df = pd.read_csv(path)

    # Sort ascending by iteration column
    iter_col = df.columns[0]
    df = df.sort_values(by=iter_col).reset_index(drop=True)

    x = df.iloc[:, 0]
    data_points = df.iloc[:, 1:]
    n_rows, n_replicates = data_points.shape

    # 2. Compute X coordinates spread between iteration i and i+1
    x_arr = x.to_numpy(dtype=float)
    if len(x_arr) > 1:
      # Step size to next iteration; last row inherits the previous step delta
      diffs = np.diff(x_arr)
      step_sizes = np.append(diffs, diffs[-1])
    else:
      step_sizes = np.array([1.0])

    if spread_mode == "random":
      offsets = np.random.uniform(
          0.0, spread_ratio, size=(n_rows, n_replicates)
      )
      x_points = x_arr[:, None] + (step_sizes[:, None] * offsets)
    else:
      # Evenly spaced fractions across [0, spread_ratio]
      fractions = (
          np.linspace(0.0, spread_ratio, n_replicates)
          if n_replicates > 1
          else np.array([0.0])
      )
      x_points = x_arr[:, None] + (step_sizes[:, None] * fractions[None, :])

    y_points = data_points.to_numpy()

    # 3. Draw individual replicate points
    ax.scatter(
        x_points.ravel(),
        y_points.ravel(),
        color=base_color,
        s=point_size,
        alpha=point_alpha,
        edgecolors="none",
        label="_nolegend_",
        zorder=2,
    )

    # 4. Bold smoothed mean curve
    row_mean = data_points.mean(axis=1)
    mean_smoothed = (
        row_mean.rolling(
            window=window_size, center=True, min_periods=1
        ).mean()
        if window_size > 1
        else row_mean
    )

    ax.plot(
        x,
        mean_smoothed,
        color=base_color,
        linewidth=line_width,
        label=series_name,
        zorder=4,
    )

    # 5. Highlight negative excursions (< 0) with darker shade
    below_zero = mean_smoothed < 0
    if below_zero.any():
      ax.fill_between(
          x,
          0,
          mean_smoothed,
          where=below_zero,
          interpolate=True,
          color=dark_tone,
          alpha=below_zero_alpha,
          hatch="//",
          label="_nolegend_",
          zorder=3,
      )

      # 6. Flag the lowest negative dip
      if highlight_trough:
        trough_idx = mean_smoothed.idxmin()
        trough_x = x.iloc[trough_idx]
        trough_y = mean_smoothed.iloc[trough_idx]

        ax.scatter(
            [trough_x],
            [trough_y],
            color=dark_tone,
            s=45,
            zorder=5,
            label="_nolegend_",
        )

        ax.annotate(
            f"{series_name} Min: {trough_y:.2f}\n@ Iter {trough_x}",
            xy=(trough_x, trough_y),
            xytext=(15, -28),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=dark_tone, lw=1.2),
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=dark_tone,
                alpha=0.95,
            ),
            fontsize=8.5,
            fontweight="bold",
            zorder=6,
        )

  ax.set_xlabel(xlabel, fontsize=12)
  ax.set_ylabel(ylabel, fontsize=12)
  ax.set_title(title, fontsize=14, fontweight="bold")
  ax.grid(True, linestyle="--", alpha=0.5)
  ax.legend(frameon=True, loc="upper right")
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



plot_iteration_scatter_envelopes(
    file_paths=files,
    colors=custom_colors,  # Blue and Green lines
    xlabel="Iteration",
    ylabel="Net Profit / Loss",
    window_size=7,
    spread_ratio=0.80,  # Points occupy 80% of the distance to next iteration
    spread_mode="linear",  # 'linear' for clean rows, 'random' for jitter
    point_size=16.0,
    point_alpha=0.30,
    below_zero_darken_factor=0.60,
    title="Iteration Metrics with Spread Replicates",
  )