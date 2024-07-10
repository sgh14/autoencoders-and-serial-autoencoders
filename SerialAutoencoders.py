import numpy as np
from Autoencoder import Autoencoder

class SerialAutoencoders:
    """
    A class representing a series of autoencoders applied sequentially.

    This class combines multiple autoencoders, where each autoencoder learns to
    reconstruct the residual error of the previous autoencoder.
    """

    def __init__(self, encoders, decoders):
        """
        Initialize the SerialAutoencoders.

        Args:
            encoders (list): A list of encoder models.
            decoders (list): A list of decoder models.
        """
        self.n_components = len(encoders)
        self.autoencoders = []
        # Create an Autoencoder instance for each encoder-decoder pair
        for encoder, decoder in zip(encoders, decoders):
            self.autoencoders.append(Autoencoder(encoder, decoder))

  
    def compile(self, *args, **kwargs):
        """
        Compile all autoencoders in the series.

        This method passes all arguments to each autoencoder's compile method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        for autoencoder in self.autoencoders:
            autoencoder.compile(*args, **kwargs)

  
    def fit(self, X, **kwargs):
        """
        Fit the series of autoencoders to the input data.

        This method trains each autoencoder sequentially, with each one learning
        to reconstruct the residual error of the previous autoencoder.

        Args:
            X (array-like): The input data to train on.
            **kwargs: Additional arguments to pass to the fit method of each autoencoder.

        Returns:
            list: A list of History objects containing training details for each autoencoder.
        """
        histories = []
        for autoencoder in self.autoencoders:
            # Train the current autoencoder
            history = autoencoder.fit(X, **kwargs)
            histories.append(history)
            # Compute the reconstruction
            X_rec = autoencoder.autoencoder(X)
            # Update X to be the residual error for the next autoencoder
            X = X - X_rec
        
        return histories

  
    def encode(self, X):
        """
        Encode the input data using all autoencoders in the series.

        Args:
            X (array-like): The input data to encode.

        Returns:
            array-like: The concatenated encoded representations from all autoencoders.
        """
        components = []
        for autoencoder in self.autoencoders:
            # Encode the current input
            component = autoencoder.encode(X)
            components.append(component)
            # Compute the reconstruction
            X_rec = autoencoder.decode(component)
            # Update X to be the residual error for the next autoencoder
            X = X - X_rec

        # Concatenate all encoded components
        X_red = np.hstack(components)

        return X_red
      

    def decode(self, X_red):
        """
        Decode the encoded data using all autoencoders in the series.

        Args:
            X_red (array-like): The encoded data to decode.

        Returns:
            array-like: The reconstructed data.
        """
        # Split the input into components for each autoencoder
        components = np.hsplit(X_red, self.n_components)
        # Decode using the first autoencoder
        X_rec = self.autoencoders[0].decode(components[0])
        # Add reconstructions from subsequent autoencoders
        for i in range(1, len(components)):
            X_rec += self.autoencoders[i].decode(components[i])

        return X_rec
