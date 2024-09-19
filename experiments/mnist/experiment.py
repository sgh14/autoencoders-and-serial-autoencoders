from os import path

from Autoencoder import Autoencoder
from experiments.mnist.plot_results import plot_original, plot_reconstruction, plot_projection, plot_interpolations, plot_history
from experiments.mnist.load_data import get_datasets
from experiments.mnist.metrics import compute_metrics
from experiments.utils import build_encoder, build_decoder


root = 'experiments/mnist/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]

datasets_train, datasets_test = get_datasets(npoints=10000, test_size=0.1, seed=123, noise=0.25)

for (X, y), title in zip(datasets_train, titles):
    plot_original(
        X, y, title, path.join(root, 'train_orig'), images_per_class=2, grid_shape=(3, 4)
    )

for (X, y), title in zip(datasets_test, titles):
    plot_original(
        X, y, title, path.join(root, 'test_orig'), images_per_class=2, grid_shape=(3, 4)
    )

for (X_train, y_train), (X_test, y_test), title in zip(datasets_train, datasets_test, titles):
    img_shape = X_train.shape[1:]
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    encoder = build_encoder(input_shape=(X_train.shape[-1],), units=128, n_components=2)
    decoder = build_decoder(output_shape=(X_train.shape[-1],), units=128, n_components=2)
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_train, epochs=300, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)

    X_train_red = autoencoder.encode(X_train)
    X_test_red = autoencoder.encode(X_test)
    X_train_rec = autoencoder.decode(X_train_red)
    X_test_rec = autoencoder.decode(X_test_red)
    
    X_train_rec = X_train_rec.numpy().reshape((X_train_rec.shape[0], *img_shape))
    X_test_rec = X_test_rec.numpy().reshape((X_test_rec.shape[0], *img_shape))

    plot_projection(X_train_red, y_train, title, path.join(root, 'train_red'))
    plot_original(
        X_train_rec, y_train, title,
        path.join(root, 'train_rec'),
        images_per_class=2, grid_shape=(3, 4)
    )
    plot_projection(X_test_red, y_test, title, path.join(root, 'test_red'))
    plot_original(
        X_test_rec, y_test, title,
        path.join(root, 'test_rec'),
        images_per_class=2, grid_shape=(3, 4)
    )
    plot_interpolations(
        X_test_red, y_test, title,
        autoencoder.decoder,
        path.join(root, 'test_interp'),
        img_shape,
        class_pairs = [(i, i+1) for i in range(0, 6, 2)],
        n_interpolations=4
    )
    plot_history(history, path.join(root, 'histories', title), log_scale=True)

    compute_metrics(X_test, X_test_red, X_test_rec, y_test, title, root)        
