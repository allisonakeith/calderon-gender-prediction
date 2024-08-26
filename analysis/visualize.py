import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


def plot_words_with_colors(word_attributions):
    fig, ax = plt.subplots(figsize=(1.5, .5))

    # Define the colors for the gradient scale
    colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # Blue, White, Red
    cmap_name = 'custom_gradient'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

    # Normalize scores to range [0, 1] for the colormap
    max_score = max(abs(score) for _, score in word_attributions)
    norm = plt.Normalize(-max_score, max_score)

    # Iterate over each word and its attribution score
    for i, (word, score) in enumerate(word_attributions):

        # Get the color from the colormap based on the normalized score
        color = cmap(norm(score))

        # Plot the word with the background color
        ax.text(i, 0, word, ha='center', va='center', fontsize=12, bbox=dict(facecolor=color, alpha=0.5))

    ax.set_axis_off()
    plt.show()
