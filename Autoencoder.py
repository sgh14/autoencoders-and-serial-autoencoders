from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input


class Autoencoder:
    """
    A class representing an Autoencoder model.

    This class combines an encoder and decoder to create a complete autoencoder,
    which can be used for dimensionality reduction and feature learning.
    """

    def __init__(self, encoder, decoder, name='autoencoder'):
        """
        Initialize the Autoencoder.

        Args:
            encoder (keras.Model): The encoder part of the autoencoder.
            decoder (keras.Model): The decoder part of the autoencoder.
        """
        # Set up the encoder
        self.encoder = encoder
        self.encoder.summary()
        
        # Set up the decoder
        self.decoder = decoder
        self.decoder.summary()

        # Construct the full autoencoder by combining encoder and decoder
        self.autoencoder = Sequential([
            Input(shape=self.encoder.input_shape[1:]),  # Input layer matching encoder's input shape
            self.encoder,
            self.decoder
        ], name=name)

        self.autoencoder.summary()

  
    def compile(self, *args, **kwargs):
        """
        Compile the autoencoder model.

        This method passes all arguments to the underlying Keras model's compile method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.autoencoder.compile(*args, **kwargs)
      

    def fit(self, X, **kwargs):
        """
        Fit the autoencoder to the input data.

        This method trains the autoencoder to reconstruct the input data.

        Args:
            X (array-like): The input data to train on.
            **kwargs: Additional arguments to pass to the fit method.

        Returns:
            history: A History object containing training details.
        """
        # Train the model to reconstruct X from X (autoencoder's nature)
        history = self.autoencoder.fit(x=X, y=X, **kwargs)
      
        return history

  
    def encode(self, X):
        """
        Encode the input data using the encoder part of the autoencoder.

        Args:
            X (array-like): The input data to encode.

        Returns:
            array-like: The encoded representation of the input data.
        """
        X_red = self.encoder(X)
      
        return X_red

  
    def decode(self, X_red):
        """
        Decode the encoded data using the decoder part of the autoencoder.

        Args:
            X_red (array-like): The encoded data to decode.

        Returns:
            array-like: The reconstructed data.
        """
        X_rec = self.decoder(X_red)
      
        return X_rec
