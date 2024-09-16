from Autoencoder import Autoencoder
from experiments.mnist.plot_results import plot_original, plot_reconstruction, plot_projection, plot_interpolations, plot_history
from experiments.mnist.load_data import get_datasets
from experiments.mnist.metrics import compute_metrics
from experiments.utils import build_encoder, build_decoder

titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]

datasets_train, datasets_test = get_datasets(npoints=10000, test_size=0.1, seed=123, noise=0.25)

# Plot datasets
plot_original(
    [datasets_train[0], datasets_train[2]],
    ['Samples without noise', 'Samples with noise'],
    'experiments/mnist/results/train_orig',
    n_classes=6,
    images_per_class=2,
    grid_shape=(3, 4)
)
plot_original(
    [datasets_test[0], datasets_test[2]],
    ['Samples without noise', 'Samples with noise'],
    'experiments/mnist/results/test_orig',
    n_classes=6,
    images_per_class=2,
    grid_shape=(3, 4)
)

datasets_train_red = []
datasets_test_red = []
datasets_train_rec = []
datasets_test_rec = []
histories = []
autoencoders = []

for (X_train, y_train), (X_test, y_test) in zip(datasets_train, datasets_test):
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    encoder = build_encoder(input_shape=(X_train.shape[-1],), units=128, n_components=2)
    decoder = build_decoder(output_shape=(X_train.shape[-1],), units=128, n_components=2)
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_train, epochs=3, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)

    X_train_red = autoencoder.encode(X_train)
    X_test_red = autoencoder.encode(X_test)
    X_train_rec = autoencoder.decode(X_train_red)
    X_test_rec = autoencoder.decode(X_test_red)
    
    X_train_rec = X_train_rec.numpy().reshape((X_train_rec.shape[0], 28, 28))
    X_test_rec = X_test_rec.numpy().reshape((X_test_rec.shape[0], 28, 28))

    datasets_train_red.append((X_train_red, y_train))
    datasets_test_red.append((X_test_red, y_test))
    datasets_train_rec.append((X_train_rec, y_train))
    datasets_test_rec.append((X_test_rec, y_test))

    autoencoders.append(autoencoder)


plot_projection(datasets_train_red, titles, 'experiments/mnist/results/train_red')
plot_reconstruction(
    datasets_train_rec,
    titles,
    'experiments/mnist/results/train_rec',
    n_classes=6,
    images_per_class=2,
    grid_shape=(3, 4)
)
plot_projection(datasets_test_red, titles, 'experiments/mnist/results/test_red')
plot_reconstruction(
    datasets_test_rec,
    titles,
    'experiments/mnist/results/test_rec',
    n_classes=6,
    images_per_class=2,
    grid_shape=(3, 4)
)

plot_interpolations(
    datasets_test_red,
    titles,
    autoencoders,
    'experiments/mnist/results/test_interp',
    (28, 28),
    class_pairs = [(i, i+1) for i in range(0, 6, 2)],
    n_interpolations=4
)

for name in ('clean-few', 'clean-many', 'noisy-few', 'noisy-many'):
    plot_history(history, 'experiments/mnist/results/histories/' + name, log_scale=True)

compute_metrics(
    datasets_test,
    datasets_test_red,
    datasets_test_rec,
    titles,
    'experiments/mnist/results',
    n_classes=6
)