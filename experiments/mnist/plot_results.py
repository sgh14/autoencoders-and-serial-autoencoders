import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

plt.style.use('experiments/science.mplstyle')
# Define your color cycle manually as a list
colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
# Create a colormap from your color cycle
cmap = ListedColormap(colors)


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
            grid_shape[0], grid_shape[1], figsize=figsize, gridspec_kw={'wspace': 0.2, 'hspace': 0.2}
        )
        axes_single = plot_images(axes_single, X, y)

        fig_single.tight_layout()
        for format in ('.pdf', '.png', '.svg'):
                fig_single.savefig(os.path.join(output_dir, title + format))
        
        plt.close(fig_single)

    
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
    subfigs = fig.subfigures(2, 1, hspace=0.05, wspace=0)
    subfigs = plot_grids(
        subfigs.ravel(),
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
    
    plt.close(fig)


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
    subfigs = fig.subfigures(2, 2, hspace=0.05, wspace=0.1)
    subfigs = plot_grids(
        subfigs.ravel(),
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

    plt.close(fig)


def plot_projection(datasets, titles, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    figsize = (6, 6)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    for ax, (X, y), title in zip(axes.flatten(), datasets, titles):
        ax.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in y])#y, cmap=cmap)
        # ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2), useMathText=True)
        ax.set_title(title)
        # Set box aspect ratio instead of axis aspect to ensure square subplots
        ax.set_box_aspect(1)

        # Create a new figure for each subplot
        fig_single, ax_single = plt.subplots(figsize=(figsize[0]/2, figsize[1]/2))
        ax_single.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in y])#y, cmap=cmap)
        ax_single.set_xlabel(r'$\Tilde{x}$')
        ax_single.set_ylabel(r'$\Tilde{y}$')
        # Set box aspect ratio instead of axis aspect to ensure square subplots
        ax_single.set_box_aspect(1)
        fig_single.tight_layout()

        # Create a list of handles and labels for the legend
        unique_y = np.unique(y)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[val], markersize=10) for val in unique_y]
        labels = [str(val) for val in unique_y]  # Adjust labels based on your case

        # Add the legend below the plot, with ncol=number of unique y values for one-row legend
        fig_single.legend(handles, labels, loc='lower center', ncol=len(unique_y)//2, bbox_to_anchor=(0.5, -0.1))

        for format in ('.pdf', '.png', '.svg'):
            fig_single.savefig(os.path.join(output_dir, title + format))
        
        plt.close(fig_single)

    axes[1, 0].set_xlabel(r'$\Tilde{x}$')
    axes[1, 1].set_xlabel(r'$\Tilde{x}$')
    axes[0, 0].set_ylabel(r'$\Tilde{y}$')
    axes[1, 0].set_ylabel(r'$\Tilde{y}$')

    # Add the legend below the plot, with ncol=number of unique y values for one-row legend
    fig.legend(handles, labels, loc='lower center', ncol=len(unique_y), bbox_to_anchor=(0.5, -0.05))


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
        for format in ('.pdf', '.png', '.svg'):
            fig.savefig(os.path.join(output_dir, key + format))
        
        plt.close(fig)


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

    interpolated_images = np.vstack(interpolated_images)

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
    subfigs = fig.subfigures(2, 2, hspace=0.05, wspace=0.1).ravel()
    for (X_red, y), title, autoencoder, subfig in zip(datasets, titles, autoencoders, subfigs):
        interpolated_images = interpolate_images(
            X_red, y, autoencoder, class_pairs, n_interpolations, image_shape
        )
        axes = subfig.subplots(grid_shape[0], grid_shape[1], gridspec_kw={'wspace': 0, 'hspace': 0})
        subfig.suptitle(title)
        axes = plot_images(axes, interpolated_images)

        fig_single, axes_single = plt.subplots(
            grid_shape[0], grid_shape[1], figsize=(figsize[0]/2, figsize[1]/2), gridspec_kw={'wspace': 0.2, 'hspace': 0}
        )
        axes_single = plot_images(axes_single, interpolated_images)

        # fig_single.tight_layout()
        for format in ('.pdf', '.png', '.svg'):
                fig_single.savefig(os.path.join(output_dir, title + format))
        
        plt.close(fig_single)

    # fig.tight_layout()
    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, 'global' + format))
    
    plt.close(fig)