from pathlib import Path
from typing import Literal, Union
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


def darken_color(color: str, factor: float = 0.65) -> tuple[float, float, float]:
  """Darkens any valid Matplotlib color by scaling its RGB channels."""
  r, g, b = mcolors.to_rgb(color)
  return (max(0.0, r * factor), max(0.0, g * factor), max(0.0, b * factor))


def normalize_to_q(
    score: pd.Series,
    q_vals: pd.Series,
    mode: Literal["match_q", "minmax", "zscore"] = "match_q",
) -> pd.Series:
  """Normalizes the agent score series to align with Q-values."""
  score_min, score_max = score.min(), score.max()
  if score_max == score_min:
    return score

  if mode == "match_q":
    q_min, q_max = q_vals.min(), q_vals.max()
    return ((score - score_min) / (score_max - score_min)) * (
        q_max - q_min
    ) + q_min
  elif mode == "minmax":
    return (score - score_min) / (score_max - score_min)
  elif mode == "zscore":
    std = score.std()
    return (score - score.mean()) / std if std > 0 else score
  return score


def plot_score_q_overlay(
    file_paths: list[str],
    colors: list[str] | None = None,
    score_col_idx: int = 1,
    q_col_idx: int = 2,
    window_size: Union[int, list[int]] = 7,
    normalize_score: bool = True,
    norm_mode: Literal["match_q", "minmax", "zscore"] = "match_q",
    show_raw: bool = False,
    show_legend: bool = True,
    y2_darken_factor: float = 0.60,
    y2_alpha: float = 0.70,
    y1_alpha: float = 1.00,
    y2_linestyle: str = "--",
    line_width: float = 2.4,
    highlight_below_zero: bool = True,
    below_zero_alpha: float = 0.35,
    highlight_trough: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str = "Agent Score vs. Q-Values",
    save_path: str | None = None,
):
  """Plots Agent Score and Q-values from multiple CSV files on a single shared canvas

  with positional list-based smoothing (Format A), color inheritance, and negative
  excursion highlighting.

  Parameters:
  ----------
  file_paths : list of str
      Paths to CSV files. Column 0 is treated as X (Iteration); columns 1 and 2
      are Score and Q-values.
  colors : list of str, optional
      Base color assigned to each file series.
  score_col_idx : int, default=1
      Positional index for the agent score column.
  q_col_idx : int, default=2
      Positional index for the Q-value column.
  window_size : int or list of int, default=7
      Smoothing window configuration (Format A):
      - Single int: applied equally to both metrics (e.g., 7).
      - List of ints [w_score, w_q]: positional window sizes for column 1 and
      column 2
        (e.g., [5, 25] smooths Score with window 5, and Q-values with window
        25).
  normalize_score : bool, default=True
      If True, rescales the agent score to match the dynamic range of Q-values.
  norm_mode : {'match_q', 'minmax', 'zscore'}, default='match_q'
      Normalization method when normalize_score is True.
  show_raw : bool, default=False
      If True, plots faint raw data traces behind the smoothed curves.
  show_legend : bool, default=True
      If True, displays the plot legend; if False, hides it.
  y2_darken_factor : float, default=0.60
      RGB channel brightness scaling for column 2 (< 1.0 is darker).
  y2_alpha : float, default=0.70
      Opacity (0 to 1) of the column 2 curve.
  y1_alpha : float, default=1.00
      Opacity (0 to 1) of the column 1 curve.
  y2_linestyle : str, default='--'
      Line style for column 2 ('--', ':', '-.').
  line_width : float, default=2.4
      Stroke width for the primary smoothed curves.
  highlight_below_zero : bool, default=True
      Shades regions below 0 in a darker variant of the respective line's color.
  below_zero_alpha : float, default=0.35
      Fill opacity for shaded negative drops.
  highlight_trough : bool, default=False
      If True, flags the deepest negative dip below zero with an annotated
      bubble.
  xlabel : str, optional
      Custom X-axis label. Defaults to column 0 header if None.
  ylabel : str, optional
      Custom Y-axis label.
  title : str, default='Agent Score vs. Q-Values'
      Plot title.
  save_path : str, optional
      File path to save the generated image.
  """
  n_files = len(file_paths)

  # Resolve window sizes using Format A (positional list or scalar)
  if isinstance(window_size, (list, tuple)):
    w_score = int(window_size[0]) if len(window_size) > 0 else 1
    w_q = int(window_size[1]) if len(window_size) > 1 else w_score
  else:
    w_score = int(window_size)
    w_q = int(window_size)

  # Fallback colors
  if colors is None:
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    default_colors = prop_cycle.by_key()["color"]
    colors = [default_colors[i % len(default_colors)] for i in range(n_files)]
  elif len(colors) < n_files:
    raise ValueError(
        f"Provided {len(colors)} colors, but need at least {n_files}."
    )

  fig, ax = plt.subplots(figsize=(11, 6))

  default_xlabel = None

  for path_str, base_color in zip(file_paths, colors):
    path = Path(path_str)
    series_name = path.stem

    df = pd.read_csv(path)

    # 1. Sort ascending by iteration (column 0)
    iter_col = df.columns[0]
    if default_xlabel is None:
      default_xlabel = iter_col
    df = df.sort_values(by=iter_col).reset_index(drop=True)

    x = df.iloc[:, 0]
    score_raw = df.iloc[:, score_col_idx]
    q_raw = df.iloc[:, q_col_idx]

    score_name = df.columns[score_col_idx]
    q_name = df.columns[q_col_idx]

    # 2. Derive darkened column 2 color
    dark_c2 = darken_color(base_color, factor=y2_darken_factor)

    # 3. Score normalization
    if normalize_score:
      score_series = normalize_to_q(score_raw, q_raw, mode=norm_mode)
      score_legend_suffix = " (Score norm)"
    else:
      score_series = score_raw
      score_legend_suffix = f" ({score_name})"

    # 4. Smoothing via Format A windows
    score_smooth = (
        score_series.rolling(window=w_score, center=True, min_periods=1).mean()
        if w_score > 1
        else score_series
    )
    q_smooth = (
        q_raw.rolling(window=w_q, center=True, min_periods=1).mean()
        if w_q > 1
        else q_raw
    )

    # 5. Optional faint raw lines
    if show_raw:
      if w_score > 1:
        ax.plot(
            x,
            score_series,
            color=base_color,
            alpha=0.20,
            linewidth=1.0,
            label="_nolegend_",
        )
      if w_q > 1:
        ax.plot(
            x,
            q_raw,
            color=dark_c2,
            alpha=0.20,
            linewidth=1.0,
            linestyle=y2_linestyle,
            label="_nolegend_",
        )

    # 6. Plot Column 1 (Score)
    ax.plot(
        x,
        score_smooth,
        color=base_color,
        alpha=y1_alpha,
        linewidth=line_width,
        label=f"{series_name}{score_legend_suffix} [w={w_score}]",
        zorder=4,
    )

    # 7. Plot Column 2 (Q-values)
    ax.plot(
        x,
        q_smooth,
        color=dark_c2,
        alpha=y2_alpha,
        linewidth=line_width * 0.9,
        linestyle=y2_linestyle,
        label=f"{series_name} ({q_name}) [w={w_q}]",
        zorder=4,
    )

    # 8. Highlight negative excursions (< 0) without a baseline rule
    if highlight_below_zero:
      for curve, c in [(score_smooth, base_color), (q_smooth, dark_c2)]:
        below_zero = curve < 0
        if below_zero.any():
          ax.fill_between(
              x,
              0,
              curve,
              where=below_zero,
              interpolate=True,
              color=c,
              alpha=below_zero_alpha,
              hatch="//",
              label="_nolegend_",
              zorder=3,
          )

      # Optional trough marker
      if highlight_trough:
        for curve, c in [(score_smooth, base_color), (q_smooth, dark_c2)]:
          if (curve < 0).any():
            trough_idx = curve.idxmin()
            trough_x = x.iloc[trough_idx]
            trough_y = curve.iloc[trough_idx]

            ax.scatter(
                [trough_x],
                [trough_y],
                color=c,
                s=35,
                zorder=5,
                label="_nolegend_",
            )
            ax.annotate(
                f"Min: {trough_y:.2f}\n@ {trough_x}",
                xy=(trough_x, trough_y),
                xytext=(12, -22),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=c, lw=1.0),
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor=c,
                    alpha=0.9,
                ),
                fontsize=8,
                fontweight="bold",
                zorder=6,
            )

  final_xlabel = xlabel if xlabel is not None else default_xlabel
  final_ylabel = (
      ylabel
      if ylabel is not None
      else ("Q-Value Scale" if normalize_score else "Metric Value")
  )

  ax.set_xlabel(final_xlabel, fontsize=12)
  ax.set_ylabel(final_ylabel, fontsize=12)
  ax.set_title(title, fontsize=14, fontweight="bold")
  ax.grid(True, linestyle="--", alpha=0.5)

  if show_legend:
    ax.legend(frameon=True, loc="upper left")

  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=300)
    print(f"Graph saved to {save_path}")

  plt.show()




# Example Usage:

files = [
    r"D:\PongRL2\Graphs\Qvalue_EpisodeAgentScore\epsilon Decay end 100000.csv",
    r"D:\PongRL2\Graphs\Qvalue_EpisodeAgentScore\epsilon Decay end 200000.csv",
    r"D:\PongRL2\Graphs\Qvalue_EpisodeAgentScore\epsilon Decay end 300000.csv"
]

files = [
    r"D:\PongRL2\Graphs\Qvalue_EpisodeAgentWins\epsilon Decay end 100000.csv",
    r"D:\PongRL2\Graphs\Qvalue_EpisodeAgentWins\epsilon Decay end 200000.csv",
    r"D:\PongRL2\Graphs\Qvalue_EpisodeAgentWins\epsilon Decay end 300000.csv"
]



  # Assign distinct colors for Column 1 vs Column 2 across files
run_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue (alpha), Green (beta)

plot_score_q_overlay(
    file_paths=files,
    colors=run_colors,
    score_col_idx=1,
    q_col_idx=2,
    window_size=[5, 250],  # Format A: [window_col1, window_col2]
    normalize_score=True,  # Normalizes score to line up over Q-values
    show_raw=False,  # Set True to show faint unsmoothed traces
    show_legend=False,  # Toggle legend display on/off
    y2_darken_factor=0.55,  # Darkens Q-value curve relative to Score
    y2_alpha=0.65,  # Transparency for Q-values
    y1_alpha=1.00,  # Opacity for Score
    y2_linestyle="--",  # Dashed line pattern for Q-values
    highlight_below_zero=True,  # Darker shaded fill for values < 0 (no baseline rule)
    title="DQN Training: Normalized Reward vs. Estimated Q-Values",
)