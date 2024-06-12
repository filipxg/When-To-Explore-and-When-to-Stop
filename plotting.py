import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
import seaborn as sns

# Inputs: xss - n lists of x values, where n is the amount of times the experiments 
def plot_binned_line_with_std(xss, yss, n_bins, y_label = "", title = "", plot_individuals = False):
    assert len(xss) == len(yss)

    mx = np.max([np.max(xs) for xs in xss])

    bin_size = (mx+1) / n_bins

    binned = []

    # What to plot on the x axis
    bins_x = np.linspace(0, mx, n_bins)
    
    for i in range(len(xss)):
        xs = xss[i]
        ys = yss[i]
        bins = [[] for _ in range(n_bins)]
        for j in range(len(xs)):
            bin_index = int(xs[j] / bin_size)
            bins[bin_index].append(ys[j])
        binned.append(bins)

    avgs = [[np.mean(bin) for bin in binned[i]] for i in range(len(xss))]
    avg = np.mean(avgs, axis=0)
    std = np.std(avgs, axis=0)

    plt.figure(figsize=(10, 6))

    if plot_individuals:
        c = 0
        for a in avgs:
            colours = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
            plt.plot(bins_x, a, linestyle=':', linewidth=1, alpha=0.7, color=colours[c])
            c += 1

    plt.plot(bins_x, avg, label='Average', color='blue')
    plt.fill_between(bins_x, avg - std, avg + std, color='blue', alpha=0.2, label='Standard Deviation')

    plt.xlabel('Environment steps')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_grid_heatmap(uncertainties, best_actions, colour_scheme = "ryg"):
    # Possible colour schemes: "ryg", "light_ryg", "hot/cold"
    height, width, _ = uncertainties.shape
    heatmap_values = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            action = best_actions[y, x]
            heatmap_values[y, x] = uncertainties[y, x, action]

    cmap = {
        "ryg": mcolors.LinearSegmentedColormap.from_list("stoplight", [(0, "green"), (0.5, "yellow"), (1, "red")]),
        "light ryg": mcolors.LinearSegmentedColormap.from_list("stoplight", [(0, "#66cdaa"), (0.5, "#fffacd"), (1, "#ff9999")]),
        "hot/cold": "coolwarm",
        }[colour_scheme]

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_values, cmap=cmap, origin='upper', interpolation='nearest')
    plt.colorbar(label='Uncertainty')
    plt.title('Uncertainties for each best action per cell')
    plt.xlabel('X')
    plt.ylabel('Y')

    for x in range(width):
        for y in range(height):
            plt.text(x, y, f'{heatmap_values[y, x]:.1f}', ha='center', va='center', color='black')

    plt.show()


def plot_barchart_rewards(reward_histories, y_label, title):
    n_agents = len(reward_histories)
    all_rewards = [reward for rewards in reward_histories for reward in rewards]

    # Separate the unique values and their corresponding counts
    unique_values, reward_counts = np.unique(np.array(all_rewards), return_counts=True)

    # Calculate the width for the bars to be adjacent
    width = np.min(np.diff(unique_values)) / n_agents if len(unique_values) > 1 else 1.0

    plt.figure()
    plt.bar(unique_values, reward_counts/n_agents, width=width, align='center')
    plt.xlabel('Unique Values')
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(unique_values)  # Ensure each unique value has a tick
    plt.tight_layout()
    plt.show()


def plot_barchart_episode_length(episode_length_histories, n_bins, y_label, title):
    # Determine the number of agents and the maximum episode length
    n_agents = len(episode_length_histories)
    max_length = max(max(lengths) for lengths in episode_length_histories)

    # Flatten the 2D list into a 1D list of all episode lengths
    all_lengths = [length for agent_lengths in episode_length_histories for length in agent_lengths]

    # Initialize the figure and axis
    fig, ax = plt.subplots()

    # Create the histogram data with the specified number of bins
    counts, bin_edges = np.histogram(all_lengths, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Divide the counts by the number of agents to get the average occurrence
    average_counts = counts / n_agents

    # Plot the bar chart
    plt.xlabel('Episode Length')
    plt.ylabel(y_label)
    plt.title(title)
    plt.bar(bin_centers, average_counts, width=(bin_edges[1] - bin_edges[0]) - 0.1, align='center')


def qtable_directions_map(qtable, map):
    """Get the best learned action & map it to arrows."""
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable = qtable.flatten()
    for idx, val in enumerate(qtable):
        qtable[int(idx)] = directions[int(val)]
    qtable_directions = qtable.reshape(len(map), len(map[0]))
    return qtable_directions

def plot_grid_statespace(state_history, optimal_moves, state_map):
    qtable_directions = qtable_directions_map(optimal_moves, state_map)
    for y in range(len(qtable_directions)):
        for x in range(len(qtable_directions[0])):
            if state_map[y][x] == 'X' or state_map[y][x] == 'G':
                qtable_directions[y, x] = ''
    state_counts = np.bincount(state_history)

    # Step 2: Normalize the visit counts
    max_count = np.max(state_counts)
    normalized_counts = state_counts / max_count
    reshaped_counts = np.reshape(normalized_counts, (len(state_map), len(state_map[0])))
    plt.figure()
    sns.heatmap(
        reshaped_counts,
        annot=qtable_directions,
        fmt="",
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")