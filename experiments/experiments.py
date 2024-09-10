from Autoencoder import Autoencoder
from experiments.plot_results import plot_original, plot_projection, plot_history
from experiments.load_data import get_datasets
from experiments.utils import build_encoder, build_decoder


datasets_train, datasets_test = get_datasets(npoints=2000, seed=123, noise=0.5)

# Plot datasets
fig, axes = plot_original(*datasets_train, 'experiments/results/swiss_roll/original/training')
fig, axes = plot_original(*datasets_test, 'experiments/results/swiss_roll/original/test')

datasets_train_red = []
datasets_test_red = []
datasets_train_rec = []
datasets_test_rec = []
histories = []

for (X_train, y_train), (X_test, y_test) in zip(datasets_train, datasets_test):
    encoder = build_encoder(input_shape=(3,), units=128, n_components=2)
    decoder = build_decoder(output_shape=(3,), units=128, n_components=2)
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_train, epochs=500, shuffle=True, batch_size=64, verbose=0)

    X_train_red = autoencoder.encode(X_train)
    X_test_red = autoencoder.encode(X_test)
    X_train_rec = autoencoder.decode(X_train_red)
    X_test_rec = autoencoder.decode(X_test_red)

    datasets_train_red.append((X_train_red, y_train))
    datasets_test_red.append((X_test_red, y_test))
    datasets_train_rec.append((X_train_rec, y_train))
    datasets_test_rec.append((X_test_rec, y_test))

fig, ax = plot_projection(*datasets_train_red, 'experiments/results/swiss_roll/AE/train_red')
fig, ax = plot_original(*datasets_train_rec, 'experiments/results/swiss_roll/AE/train_rec')
fig, axes = plot_projection(*datasets_test_red, 'experiments/results/swiss_roll/AE/test_red')
fig, axes = plot_original(*datasets_test_rec, 'experiments/results/swiss_roll/AE/test_rec')

for i, name in ('clean-few', 'clean-many', 'noisy-few', 'noisy-many'):
    plot_history(history, 'experiments/results/swiss_roll/AE/histories/' + name)
