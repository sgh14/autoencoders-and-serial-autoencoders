from Autoencoder import Autoencoder
from experiments.mnist.plot_results import plot_original, plot_reconstruction, plot_projection, plot_history
from experiments.mnist.load_data import get_datasets
from experiments.utils import build_encoder, build_decoder


datasets_train, datasets_test = get_datasets(npoints=10000, test_size=0.1, seed=123, noise=0.5)

print(len(datasets_train), len(datasets_test))
# Plot datasets
fig, axes = plot_original(
    *datasets_train,
    'experiments/mnist/results/train_orig',
    num_classes=6,
    images_per_class=2,
    grid_shape=(3, 4)
)
fig, axes = plot_original(
    *datasets_test,
    'experiments/mnist/results/test_orig',
    num_classes=6,
    images_per_class=2,
    grid_shape=(3, 4)
)

datasets_train_red = []
datasets_test_red = []
datasets_train_rec = []
datasets_test_rec = []
histories = []

for (X_train, y_train), (X_test, y_test) in zip(datasets_train, datasets_test):
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    encoder = build_encoder(input_shape=(3,), units=128, n_components=2)
    decoder = build_decoder(output_shape=(3,), units=128, n_components=2)
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_train, epochs=500, validation_split=0.1, shuffle=True, batch_size=64, verbose=0)

    X_train_red = autoencoder.encode(X_train)
    X_test_red = autoencoder.encode(X_test)
    X_train_rec = autoencoder.decode(X_train_red)
    X_test_rec = autoencoder.decode(X_test_red)
    
    X_train_rec = X_train_rec.reshape((X_train_rec.shape[0], 28, 28))
    X_test_rec = X_test_rec.reshape((X_test_rec.shape[0], 28, 28))

    datasets_train_red.append((X_train_red, y_train))
    datasets_test_red.append((X_test_red, y_test))
    datasets_train_rec.append((X_train_rec, y_train))
    datasets_test_rec.append((X_test_rec, y_test))

fig, ax = plot_projection(
    *datasets_train_red,
    'experiments/mnist/results/train_red',
    num_classes=6,
    images_per_class=2,
    grid_shape=(3, 4)
)
fig, ax = plot_projection(
    *datasets_train_rec,
    'experiments/mnist/results/train_rec',
    num_classes=6,
    images_per_class=2,
    grid_shape=(3, 4)
)
fig, ax = plot_projection(
    *datasets_test_red,
    'experiments/mnist/results/test_red',
    num_classes=6,
    images_per_class=2,
    grid_shape=(3, 4)
)
fig, ax = plot_projection(
    *datasets_test_rec,
    'experiments/mnist/results/test_rec',
    num_classes=6,
    images_per_class=2,
    grid_shape=(3, 4)
)

for name in ('clean-few', 'clean-many', 'noisy-few', 'noisy-many'):
    plot_history(history, 'experiments/mnist/results/histories/' + name, log_scale=True)
