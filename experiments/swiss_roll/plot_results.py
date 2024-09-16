import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# Define custom style
plt.style.use('default')  # Reset to default style
mpl.rcParams.update({
    # Use serif fonts
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # Specify a specific serif font
    'font.size': 15,
    'axes.titlesize': 'medium',
    'axes.labelsize': 'medium',
    'mathtext.fontset': 'dejavuserif',

    # Use LaTeX for math formatting
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
})


def my_colormap2D(x, y):
    # Define colors in RGB
    bottom_left = (0.5, 0, 0.5) # dark magenta
    bottom_right = (0, 0.5, 0.5) # dark cyan
    top_left = (1, 0, 1) # magenta
    top_right = (0, 1, 1) # cyan

    # Calculate the RGB values based on interpolation
    top_color = np.array(top_left) * (1 - x) + np.array(top_right) * x
    bottom_color = np.array(bottom_left) * (1 - x) + np.array(bottom_right) * x

    return top_color * (1 - y) + bottom_color * y


def plot_original(dataset_small, dataset, dataset_noisy_small, dataset_noisy, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    figsize = (6, 6)
    fig, axes = plt.subplots(2, 2, figsize=figsize, subplot_kw={"projection": "3d"})

    datasets = [dataset_small, dataset, dataset_noisy_small, dataset_noisy]
    titles = [
        'Few samples without noise',
        'Many samples without noise',
        'Few samples with noise',
        'Many samples with noise'
    ]
    for ax, (data, label), title in zip(axes.flatten(), datasets, titles):
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=label, alpha=1)
        ax.set_title(title)
        ax.set_xlabel('$x$')
        ax.set_xlim([-13, 13])
        ax.set_ylabel('$y$')
        ax.set_ylim([-3, 23])
        ax.set_zlabel('$z$')
        ax.set_zlim([-13, 13])
        ax.view_init(15, -72)

        # Create a new figure for each subplot
        fig_single, ax_single = plt.subplots(figsize=(figsize[0]/2, figsize[1]/2), subplot_kw={"projection": "3d"})
        ax_single.scatter(data[:, 0], data[:, 1], data[:, 2], c=label, alpha=1)
        ax_single.set_xlabel('$x$')
        ax_single.set_xlim([-13, 13])
        ax_single.set_ylabel('$y$')
        ax_single.set_ylim([-3, 23])
        ax_single.set_zlabel('$z$')
        ax_single.set_zlim([-13, 13])
        ax_single.view_init(15, -72)
        fig_single.tight_layout()

        for format in ('.pdf', '.png', '.svg'):
            fig_single.savefig(os.path.join(output_dir, title + format))

        plt.close(fig_single)

    fig.tight_layout()
    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, 'global' + format))
    
    plt.close(fig)


def plot_projection(dataset_small, dataset, dataset_noisy_small, dataset_noisy, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    figsize = (6, 6)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    datasets = [dataset_small, dataset, dataset_noisy_small, dataset_noisy]
    titles = [
        'Few samples without noise',
        'Many samples without noise',
        'Few samples with noise',
        'Many samples with noise'
    ]

    for ax, (data, label), title in zip(axes.flatten(), datasets, titles):
        ax.scatter(data[:, 0], data[:, 1], c=label)
        # ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2), useMathText=True)
        ax.set_title(title)
        ax.set_box_aspect(1)

        # Create a new figure for each subplot
        fig_single, ax_single = plt.subplots(figsize=(figsize[0]/2, figsize[1]/2))
        ax_single.scatter(data[:, 0], data[:, 1], c=label)
        ax_single.set_xlabel(r'$\Tilde{x}$')
        ax_single.set_ylabel(r'$\Tilde{y}$')
        ax_single.set_box_aspect(1)
        fig_single.tight_layout()

        for format in ('.pdf', '.png', '.svg'):
            fig_single.savefig(os.path.join(output_dir, title + format))
        
        plt.close(fig_single)

    axes[1, 0].set_xlabel(r'$\Tilde{x}$')
    axes[1, 1].set_xlabel(r'$\Tilde{x}$')
    axes[0, 0].set_ylabel(r'$\Tilde{y}$')
    axes[1, 0].set_ylabel(r'$\Tilde{y}$')
    
    for format in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(output_dir, 'global.' + format))
    
    plt.close(fig)


def plot_history(history, output_dir, log_scale=False):
    os.makedirs(output_dir, exist_ok=True)

    h = history.history
    keys = [key for key in h.keys() if not key.startswith('val_')]
    for key in keys:
        y = np.array([h[key], h['val_' + key]])
        fig, ax = plt.subplots()
        if log_scale:
            ax.semilogy(y[0], label='Training')
            ax.semilogy(y[1], label='Validation')
        else:
            ax.plot(y[0], label='Training')
            ax.plot(y[1], label='Validation')

        ax.set_ylabel(key)
        ax.set_xlabel('Epoch')
        ax.legend()
        for format in ('pdf', 'png', 'svg'):
            fig.savefig(os.path.join(output_dir, key + format))
        
        plt.close(fig)