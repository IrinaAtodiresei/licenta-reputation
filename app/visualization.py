import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ============================================================
# 1. WORD CONTRIBUTION BAR CHART
# ============================================================
def plot_word_contributions(tokens, contributions, title, xlabel):
    """
    Creates a horizontal bar chart showing word-level sentiment contributions.
    Positive = green, Negative = red.
    """

    plt.figure(figsize=(10, 5))
    colors = ["green" if c > 0 else "red" for c in contributions]

    y_pos = np.arange(len(tokens))

    plt.barh(y_pos, contributions, color=colors)
    plt.yticks(y_pos, tokens)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    return plt.gcf()


# ============================================================
# 2. TRANSFORMER ATTENTION HEATMAP
# ============================================================
def plot_transformer_attention_heatmap(tokens, attention_matrix, title="Transformer Attention Heatmap"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        square=True,
        cbar=True
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return plt.gcf()

