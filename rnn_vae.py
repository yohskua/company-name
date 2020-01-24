from base_vaes import VariationalAutoencoder as vae
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from sklearn.model_selection import train_test_split


class RnnVae(vae):
    def __init__(self, X, GO, EOS, *args, **kwargs):
        self.GO = GO
        self.EOS = EOS
        self.X = X
        self.max_sequence_length = None
        self.X_in_decoder = None
        self.X_in_encoder = None
        self.dataset_size = None
        self.characters = None
        self.char_index = None
        self.index_char = None
        self.vocabulary_size = None
        self.model = None
        self._set_data_properties_attributes()
        self._construct_data_set()

        super(RnnVae, self).__init__(
            self.X_tr, self.X_te, None, None, None, flatten=False, *args, **kwargs
        )
        self.design_and_compile_full_model()

    def _set_data_properties_attributes(self):
        # max_sequence_length is longest sequence + GO term
        # at beginning if as input of decoder or
        # longest sequence + EOS term at end if as input of encoder / output of decoder
        self.max_sequence_length = max([len(sequence) for sequence in self.X]) + 1
        self.X_in_decoder = list(map(lambda token: self.GO + token, self.X))
        self.X_in_encoder = list(
            map(
                lambda token: token
                + self.EOS * (self.max_sequence_length - len(token)),
                self.X,
            )
        )

        self.dataset_size = len(self.X)
        self.characters = list(set("".join(self.X)))
        assert self.GO not in self.characters and self.EOS not in self.characters
        self.characters = sorted(self.characters + [self.GO, self.EOS])
        self.char_index = dict((c, i) for i, c in enumerate(self.characters))
        self.index_char = dict((i, c) for i, c in enumerate(self.characters))
        self.vocabulary_size = len(self.characters)

        print("Number of samples:", self.dataset_size)
        print("Number of unique tokens:", self.vocabulary_size)
        print("Max sequence length:", self.max_sequence_length)

    def _one_hot_encode(self, word):
        input = np.zeros(
            (1, self.max_sequence_length, self.vocabulary_size), dtype="float32"
        )
        for t, char in enumerate(word):
            input[0, t, self.char_index[char]] = 1.0
        return input

    def _construct_data_set(self):
        encoder_input = np.zeros(
            (self.dataset_size, self.max_sequence_length, self.vocabulary_size),
            dtype="float32",
        )
        decoder_input = np.zeros(
            (self.dataset_size, self.max_sequence_length, self.vocabulary_size),
            dtype="float32",
        )

        for i, X_i in enumerate(self.X_in_encoder):
            encoder_input[i] = self._one_hot_encode(X_i)
        for i, X_i in enumerate(self.X_in_decoder):
            decoder_input[i] = self._one_hot_encode(X_i)

        (
            encoder_input_tr,
            encoder_input_te,
            decoder_input_tr,
            decoder_input_te,
        ) = train_test_split(
            encoder_input, decoder_input, random_state=42, test_size=0.25
        )
        self.X_tr = [encoder_input_tr, decoder_input_tr]
        self.X_te = [encoder_input_te, decoder_input_te]

    def design_and_compile_encoder(self):
        encoder_input = Input(shape=(self.max_sequence_length, self.vocabulary_size))
        _, h_T, _ = LSTM(
            self.intermediate_dim, return_state=True, return_sequences=True
        )(encoder_input)
        z_mean = Dense(self.latent_dim)(h_T)
        z_log_var = Dense(self.latent_dim)(h_T)

        return Model(
            inputs=encoder_input, outputs=[z_mean, z_log_var], name="rnn_encoder"
        )

    def design_and_compile_decoder(self):
        rnn_state_input = Input(shape=(self.latent_dim,))
        decoder_input = Input(shape=(None, self.vocabulary_size))
        decoder_lstm = LSTM(self.latent_dim, return_state=True, return_sequences=True)
        # send one vector as both c_t and h_t so as not to have to define yet another input layer
        decoder_all_hdec, rnn_state_last, _ = decoder_lstm(
            decoder_input, initial_state=[rnn_state_input, rnn_state_input]
        )
        decoder_output = Dense(self.vocabulary_size, activation="softmax")(
            decoder_all_hdec
        )

        return Model(
            [rnn_state_input, decoder_input],
            [rnn_state_last, decoder_output],
            name="rnn_decoder",
        )

    def design_and_compile_full_model(self):
        self.encoder = self.design_and_compile_encoder()
        self.sampler = self.design_and_compile_sampler()
        self.decoder = self.design_and_compile_decoder()
        # encoder input right there
        # shape change it after
        encoder_input = Input(
            shape=(self.max_sequence_length, self.vocabulary_size), name="encoder_input"
        )
        # None dim, prepare case where 1 timestep long sequences are sent
        decoder_input = Input(shape=(None, self.vocabulary_size), name="decoder_input")

        z_mean, z_log_var = self.encoder(encoder_input)
        z = self.sampler([z_mean, z_log_var])
        rnn_state_last, x_decoded_mean = self.decoder([z, decoder_input])
        self.model = Model([encoder_input, decoder_input], x_decoded_mean)

        x = encoder_input
        xent_loss = self.original_dim * metrics.binary_crossentropy(
            K.flatten(x), K.flatten(x_decoded_mean)
        )
        kl_loss = -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
        )
        vae_loss = K.mean(xent_loss + kl_loss)

        self.model.add_loss(vae_loss)
        self.model.compile(optimizer="adam")

    def decode_latent(self, z_latent):
        state_value = z_latent
        decoder_input = np.zeros((1, 1, self.vocabulary_size))
        decoder_input[0, 0, self.char_index[self.GO]] = 1.0
        decoded_sentence = ""
        while len(decoded_sentence) <= self.max_sequence_length:
            rnn_state_last, output_token = self.decoder.predict(
                [state_value, decoder_input]
            )
            state_value = rnn_state_last
            sampled_token_index = np.argmax(output_token[0, 0, :])
            sampled_char = self.index_char[sampled_token_index]
            if sampled_char == self.EOS:
                break
            decoded_sentence += sampled_char
            decoder_input = np.zeros((1, 1, self.vocabulary_size))
            decoder_input[0, 0, sampled_token_index] = 1.0
        return decoded_sentence

    def sample_prediction(self):
        # problem is now we always sample around the same point: 0 mean and 1 std.
        sampled_latent_vector = np.random.normal(size=(1, self.latent_dim))
        return self.decode_latent(sampled_latent_vector)

    def sample_around(self, word):
        # model has been trained to encode and reconstruct EOS tokens at the end of word
        word = word + self.EOS * (self.max_sequence_length - len(word))
        encoder_input = self._one_hot_encode(word)
        z_mean, z_log_var = self.encoder.predict(encoder_input)
        z = self.sampler.predict([z_mean, z_log_var])
        return self.decode_latent(z)
