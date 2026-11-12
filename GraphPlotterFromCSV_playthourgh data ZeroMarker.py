from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


def darken_color(color: str, factor: float = 0.65) -> tuple[float, float, float]:
  """Darkens any valid Matplotlib color (hex or named) by scaling its RGB channels."""
  r, g, b = mcolors.to_rgb(color)
  return (max(0.0, r * factor), max(0.0, g * factor), max(0.0, b * factor))


def plot_iteration_below_zero(
    file_paths: list[str],
    colors: list[str] | None = None,
    xlabel: str = "Iteration",
    ylabel: str = "Metric Value",
    window_size: int = 5,
    envelope_smooth_window: int = 5,
    band_alpha: float = 0.18,
    below_zero_darken_factor: float = 0.65,
    below_zero_alpha: float = 0.45,
    line_width: float = 2.5,
    highlight_trough: bool = True,
    title: str = "Iteration Performance with Negative Value Highlights",
    save_path: str | None = None,
):
  """Plots iteration data with rolling envelopes, shading negative regions (<0)

  using a darkened shade of each individual series color.

  Parameters:
  ----------
  file_paths : list of str
      Paths to CSV files. Column 0 is the iteration index; remaining columns are
      replicates.
  colors : list of str, optional
      Base colors for each series. Defaults to Matplotlib cycle if None.
  xlabel : str, default='Iteration'
      X-axis label.
  ylabel : str, default='Metric Value'
      Y-axis label.
  window_size : int, default=5
      Smoothing window for the bold mean curve.
  envelope_smooth_window : int, default=5
      Smoothing window for the min/max band edges.
  band_alpha : float, default=0.18
      Transparency of the standard fluctuation envelope band.
  below_zero_darken_factor : float, default=0.65
      Multiplier to darken the base color for negative drops (< 1.0 is darker).
  below_zero_alpha : float, default=0.45
      Opacity of the shaded area when values fall below 0.
  line_width : float, default=2.5
      Stroke width for the bold mean curve.
  highlight_trough : bool, default=True
      If True, annotates the lowest negative point for each series.
  title : str, default='Iteration Performance with Negative Value Highlights'
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

  # Baseline at zero
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

    # Derive a darker variant of this specific line's color
    dark_tone = darken_color(base_color, factor=below_zero_darken_factor)

    df = pd.read_csv(path)

    # Sort ascending by the iteration column
    iter_col = df.columns[0]
    df = df.sort_values(by=iter_col).reset_index(drop=True)

    x = df.iloc[:, 0]
    data_points = df.iloc[:, 1:]

    # Row-wise statistics across replicates
    row_mean = data_points.mean(axis=1)
    row_min = data_points.min(axis=1)
    row_max = data_points.max(axis=1)

    # Apply rolling averages
    mean_smoothed = (
        row_mean.rolling(
            window=window_size, center=True, min_periods=1
        ).mean()
        if window_size > 1
        else row_mean
    )
    min_smoothed = (
        row_min.rolling(
            window=envelope_smooth_window, center=True, min_periods=1
        ).mean()
        if envelope_smooth_window > 1
        else row_min
    )
    max_smoothed = (
        row_max.rolling(
            window=envelope_smooth_window, center=True, min_periods=1
        ).mean()
        if envelope_smooth_window > 1
        else row_max
    )

    # 1. Standard fluctuation ribbon (lighter opacity)
    ax.fill_between(
        x,
        min_smoothed,
        max_smoothed,
        color=base_color,
        alpha=band_alpha,
        linewidth=0,
        label="_nolegend_",
    )

    # 2. Main smoothed trendline
    ax.plot(
        x,
        mean_smoothed,
        color=base_color,
        linewidth=line_width,
        label=series_name,
        zorder=4,
    )

    # 3. Highlight excursions below 0 in the darkened color
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
          hatch="//",  # Hatched lines give additional contrast against the base envelope
          label="_nolegend_",
          zorder=3,
      )

      # 4. Marker and callout bubble for the deepest negative dip
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
            arrowprops=dict(
                arrowstyle="->",
                color=dark_tone,
                lw=1.2,
            ),
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



plot_iteration_below_zero(
    file_paths=files,
    colors=custom_colors,  # Blue and Green lines
    xlabel="Iteration",
    ylabel="Net Profit / Loss",
    window_size=7,
    envelope_smooth_window=50,
    below_zero_darken_factor=0.60,  # 0.60 scales RGB channels to 60% brightness (40% darker)
    below_zero_alpha=0.45,  # Higher alpha ensures the darker region stands out
    highlight_trough=False,
    title="Training Metrics with Darkened Deficit Shading",
  )