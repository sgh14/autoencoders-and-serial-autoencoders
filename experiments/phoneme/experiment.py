import time
import numpy as np
import tensorflow as tf
import os
import random
from os import path
import h5py

from Autoencoder import Autoencoder
from experiments.phoneme.load_data import get_datasets
from experiments.utils import build_seq_encoder, build_seq_decoder

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ENSURE REPRODUCIBILITY
seed = 123
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # force the use of CPU


root = '/scratch/sgarcia/tfm/AE/experiments/phoneme/results'
titles = [
    'Few samples without noise',
    'Many samples without noise',
    'Few samples with noise',
    'Many samples with noise'
]

datasets_train, datasets_test = get_datasets(test_size=0.2, seed=seed, noise=0.25)

for (X_train, y_train), (X_test, y_test), title in zip(datasets_train, datasets_test, titles):
    print(title)
    output_dir = path.join(root, title)
    os.makedirs(output_dir, exist_ok=True)
    encoder = build_seq_encoder(input_shape=X_train.shape[1:], filters=8, n_components=2, zero_padding=0, use_bn=False)
    decoder = build_seq_decoder(output_shape=X_train.shape[1:], filters=8, n_components=2, cropping=0)
    autoencoder = Autoencoder(encoder, decoder)
    tic = time.perf_counter()
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    history = autoencoder.fit(X_train, epochs=100, validation_split=0.1, shuffle=False, batch_size=64, verbose=0)

    X_train_red = autoencoder.encode(X_train)
    tac = time.perf_counter()
    X_test_red = autoencoder.encode(X_test)
    toc = time.perf_counter()
    X_train_rec = autoencoder.decode(X_train_red).numpy()
    X_test_rec = autoencoder.decode(X_test_red).numpy()

    autoencoder.encoder.save(path.join(output_dir, 'encoder.keras'))
    autoencoder.decoder.save(path.join(output_dir, 'decoder.keras'))
    with h5py.File(path.join(output_dir, 'history.h5'), 'w') as file:
        for key, value in history.history.items():
            file.create_dataset(key, data=value)
    
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_train_rec = X_train_rec.reshape(X_train_rec.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_test_rec = X_test_rec.reshape(X_test_rec.shape[0], -1)
    with h5py.File(path.join(output_dir, 'results.h5'), "w") as file:
        file.create_dataset("X_train", data=X_train, compression='gzip')
        file.create_dataset("X_train_red", data=X_train_red, compression='gzip')
        file.create_dataset("X_train_rec", data=X_train_rec, compression='gzip')
        file.create_dataset("y_train", data=y_train, compression='gzip')
        file.create_dataset("X_test", data=X_test, compression='gzip')
        file.create_dataset("X_test_red", data=X_test_red, compression='gzip')
        file.create_dataset("X_test_rec", data=X_test_rec, compression='gzip')
        file.create_dataset("y_test", data=y_test, compression='gzip')

    time_in_sample = tac - tic
    time_out_of_sample = toc - tac
    times = np.array([time_in_sample, time_out_of_sample])
    np.savetxt(path.join(output_dir, 'times.txt'), times)