import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Inputs: xss - n lists of x values, where n is the amount of times the experiments 
def plot_binned_line_with_std(xss, yss, n_bins, y_label = "", title = "", plot_individuals = False):
    assert len(xss) == len(yss)

    bin_size = (np.max(xss)+1) / n_bins

    binned = []

    # What to plot on the x axis
    bins_x = np.linspace(0, np.max(xss), n_bins)
    
    for i in range(len(xss)):
        xs = xss[i]
        ys = yss[i]
        bins = [[] for _ in range(n_bins)]
        for j in range(len(xs)):
            bin_index = int(xs[j] / bin_size)
            bins[bin_index].append(ys[j])
        binned.append(np.array(bins))

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
