import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

# Define smoothing parameter: n steps before + current point + n steps after
N_STEPS = 5
WINDOW_SIZE = 1 * N_STEPS + 1

# List of files, custom labels, and assigned colors
filenames =[
    "modelWeights__Nov03_22-49-49_animatePlayedPongScores.csv",
    "modelWeights__Nov04_15-56-28_animatePlayedPongScores.csv",
    "modelWeights__Nov05_07-44-49_animatePlayedPongScores.csv"
]
mainPath = Path(r"D:\PongRL2")

experiments = [
    {"file": mainPath / filenames[0], "label": "Model doubleDQN decay end 100000", "color": "#1f77b4"},
    {"file": mainPath / filenames[1], "label": "Model doubleDQN decay end 200000", "color": "#ff7f0e"},
    {"file": mainPath / filenames[2], "label": "Model doubleDQN decay end 300000", "color": "#2ca02c"},
]

fig, ax = plt.subplots(figsize=(10, 6))

for exp in experiments:
    df = pd.read_csv(exp["file"])

    # 1. Cast column 0 to numeric and sort
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0])
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)

    # 2. Extract columns
    iterations = df.iloc[:, 0]
    avg_score = df.iloc[:, 1]
    raw_scores = df.iloc[:, 2:]

    # 3. Calculate raw min and max per iteration
    min_score = raw_scores.min(axis=1)
    max_score = raw_scores.max(axis=1)

    # 4. Smooth average, min, and max using centered moving average
    smooth_avg = avg_score.rolling(
        window=WINDOW_SIZE, center=True, min_periods=1
    ).mean()
    smooth_min = min_score.rolling(
        window=WINDOW_SIZE, center=True, min_periods=1
    ).mean()
    smooth_max = max_score.rolling(
        window=WINDOW_SIZE, center=True, min_periods=1
    ).mean()

    # 5. Plot smoothed trendline and envelope
    ax.plot(
        iterations,
        smooth_avg,
        label=exp["label"],
        color=exp["color"],
        linewidth=2,
    )
    ax.fill_between(
        iterations,
        smooth_min,
        smooth_max,
        color=exp["color"],
        alpha=0.2,
    )

ax.set_title(
    f"Iteration vs. Score (Smoothed: ±{N_STEPS} Steps)",
    fontsize=14,
)
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="best", frameon=True)

plt.tight_layout()
plt.show()