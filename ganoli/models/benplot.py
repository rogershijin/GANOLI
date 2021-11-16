import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

def convert_to_one_hot(col):
    vals = list(set(col))
    one_hot_vars = np.array([vals.index(x) for x in col], dtype=np.float32)
    return one_hot_vars, vals

def plot_lowdim_rep(low_dim, labels, xlabel="UMAP 1", ylabel="UMAP 2",
    min_samples=100, figname=None, cbar_label=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))  # setup the plot
    cmap = plt.cm.jet  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    tag, tag_names = convert_to_one_hot(labels)
    order = np.argsort(tag_names)
    tag_names = np.array(tag_names)[order]
    tag = np.array([list(order).index(int(x)) for x in tag])
    good_tags = [np.sum(tag == i) > min_samples for i in range(len(tag_names))]
    tag_names = np.array(tag_names)[good_tags]
    good_idxs = np.array([good_tags[int(tag[i])] for i in range(len(tag))])
    tag = tag[good_idxs]
    tag, _ = convert_to_one_hot(tag)
    bounds = np.linspace(0, len(tag_names), len(tag_names)+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    scat = ax.scatter(low_dim[good_idxs, 0], low_dim[good_idxs, 1],
                      c=tag, marker='+', alpha=0.6, #s=np.random.randint(100, 500, 20),
                      cmap=cmap, norm=norm)
    plt.xlabel(xlabel, fontsize=48)
    plt.ylabel(ylabel, fontsize=48)
    plt.xticks([])
    plt.yticks([])
    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.95, 0.15, 0.03, 0.7])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
        spacing='proportional', ticks=bounds+0.5,#boundaries=bounds,
                                   format='%1i')
    cb.ax.set_yticklabels(tag_names, fontsize=24)
    if cbar_label is not None:
        cb.ax.set_ylabel(cbar_label, fontsize=32)
    if figname is not None:
        plt.savefig("results/{}.pdf".format(figname), dpi=300, bbox_inches='tight')