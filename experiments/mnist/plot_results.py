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


# Function to sample 2 images per class from the dataset
def sample_images_per_class(X, y, n_classes=6, images_per_class=2):
    selected_images = []
    selected_labels = []
    
    for class_label in range(n_classes):
        class_indices = np.where(y == class_label)[0]
        selected_indices = class_indices[:images_per_class]
        selected_images.extend(X[selected_indices])
        selected_labels.extend(y[selected_indices])
    
    return np.array(selected_images), np.array(selected_labels)


def plot_images(axes, X, y=[]):
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X[i], cmap='gray')
        if len(y) > 0:
            ax.set_title(y[i])

        ax.axis('off')
    
    return axes


def plot_grids(
    subfigs,
    datasets,
    titles,
    output_dir,
    n_classes=6,
    images_per_class=2,
    grid_shape=(3, 4),
    figsize=(3, 3)
):
    os.makedirs(output_dir, exist_ok=True)
    
    for subfig, (X, y), title in zip(subfigs, datasets, titles):
        X, y = sample_images_per_class(X, y, n_classes, images_per_class)
        axes = subfig.subplots(grid_shape[0], grid_shape[1], gridspec_kw={'wspace': 0, 'hspace': 0})
        subfig.suptitle(title)
        axes = plot_images(axes, X, y)

        fig_single, axes_single = plt.subplots(
            grid_shape[0], grid_shape[1], figsize=figsize
        )
        axes_single = plot_images(axes_single, X, y)

        fig_single.tight_layout()
        for format in ('.pdf', '.png', '.svg'):
                fig_single.savefig(os.path.join(output_dir, title + format))

    
    return subfigs


def plot_original(
    datasets,
    titles,
    output_dir,
    n_classes=6,
    images_per_class=2,
    grid_shape=(3, 4)
):
    os.makedirs(output_dir, exist_ok=True)
    figsize = (3, 6)
    # Create a main figure
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    # Create two subfigures for clean and noisy images
    subfigs = fig.subfigures(2, 1, hspace=0.1, wspace=0)
    subfigs = plot_grids(
        subfigs,
        datasets,
        titles,
        output_dir,
        n_classes,
        images_per_class,
        grid_shape,
        (figsize[0], figsize[1]/2)
    )
    
    # fig.tight_layout()
    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, 'global' + format))


def plot_reconstruction(
    datasets,
    titles,
    output_dir,
    n_classes=6,
    images_per_class=2,
    grid_shape=(3, 4)
):
    os.makedirs(output_dir, exist_ok=True)
    figsize = (6, 6)
    # Create a main figure
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    # Create two subfigures for clean and noisy images
    subfigs = fig.subfigures(2, 2, hspace=0.1, wspace=0)
    subfigs = plot_grids(
        subfigs,
        datasets,
        titles,
        output_dir,
        n_classes,
        images_per_class,
        grid_shape,
        (figsize[0]/2, figsize[1]/2)
    )

    # fig.tight_layout()
    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, 'global' + format))


def plot_projection(datasets, titles, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    figsize = (6, 6)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, (X, y), title in zip(axes.flatten(), datasets, titles):
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='Spectral')
        # ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2), useMathText=True)
        ax.set_title(title)

        # Create a new figure for each subplot
        fig_single, ax_single = plt.subplots(figsize=(figsize[0]/2, figsize[1]/2))
        ax_single.scatter(X[:, 0], X[:, 1], c=y, cmap='Spectral')
        ax_single.set_aspect('equal')
        ax_single.set_xlabel(r'$\Tilde{x}$')
        ax_single.set_ylabel(r'$\Tilde{y}$')
        fig_single.tight_layout()

        for format in ('.pdf', '.png', '.svg'):
            fig_single.savefig(os.path.join(output_dir, title + format))

    axes[1, 0].set_xlabel(r'$\Tilde{x}$')
    axes[1, 1].set_xlabel(r'$\Tilde{x}$')
    axes[0, 0].set_ylabel(r'$\Tilde{y}$')
    axes[1, 0].set_ylabel(r'$\Tilde{y}$')
    # Set all axes to be square
    for ax in axes.flatten():
        ax.set_aspect('equal')

    fig.tight_layout()
    for format in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(output_dir, 'global.' + format))


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
        fig.savefig(os.path.join(output_dir, key + '.png'))


# Helper function to compute class centroids in latent space
def compute_centroids(X_red, y):
    n_classes = len(np.unique(y))
    centroids = [np.mean(X_red[y == i], axis=0) for i in range(n_classes)]

    return np.array(centroids)

# Helper function to interpolate between two centroids
def interpolate(x1, x2, n_interpolations=4):
    alphas = np.linspace(0, 1, n_interpolations)
    interpolations = [(1 - alpha) * x1 + alpha * x2 for alpha in alphas]
    
    return np.array(interpolations)


def interpolate_images(X_red, y, autoencoder, class_pairs, n_interpolations, image_shape):
    centroids = compute_centroids(X_red, y)
    interpolated_images = []
    for class1, class2 in class_pairs:
        centroid1 = centroids[class1]
        centroid2 = centroids[class2]
        interpolations = interpolate(centroid1, centroid2, n_interpolations)
        interpolated_images.append(
            autoencoder.decode(interpolations).numpy().reshape(-1, *image_shape)
        )

    interpolated_images = np.stack(interpolated_images, axis=0)

    return interpolated_images


# Function to generate and plot interpolations between two classes
def plot_interpolations(
    datasets,
    titles,
    autoencoders,
    output_dir,
    image_shape,
    class_pairs = [(i, i+1) for i in range(0, 6, 2)],
    n_interpolations=4
):
    os.makedirs(output_dir, exist_ok=True)
    figsize = (6, 6)
    grid_shape = (len(class_pairs), n_interpolations)
    # Create a main figure
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    # Create two subfigures for clean and noisy images
    subfigs = fig.subfigures(2, 2, hspace=0.1, wspace=0)
    for (X_red, y), title, autoencoder, subfig in zip(datasets, titles, autoencoders, subfigs):
        interpolated_images = interpolate_images(
            X_red, y, autoencoder, class_pairs, n_interpolations, image_shape
        )
        axes = subfig.subplots(grid_shape[0], grid_shape[1], gridspec_kw={'wspace': 0, 'hspace': 0})
        subfig.suptitle(title)
        axes = plot_images(axes, interpolated_images)

        fig_single, axes_single = plt.subplots(
            grid_shape[0], grid_shape[1], figsize=(figsize[0]/2, figsize[1]/2)
        )
        axes_single = plot_images(axes_single, interpolated_images)

        fig_single.tight_layout()
        for format in ('.pdf', '.png', '.svg'):
                fig_single.savefig(os.path.join(output_dir, title + format))

    # fig.tight_layout()
    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, 'global' + format))