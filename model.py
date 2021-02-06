import random
import numpy as np

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.losses import mse, binary_crossentropy

from sklearn.metrics import precision_recall_fscore_support

class FAKE_NEWS_DETECTOR():
    def __init__(self, input_d, domain_emb_d, latent_d, lambda1, lambda2, lambda3):
        self.input_shape = input_d
        self.latent_dim = latent_d
        self.no_domains = domain_emb_d
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # build and compile the domain-specific discriminator
        self.discriminator_specific = self.build_discriminator(name = 'specific_discriminator')
        self.discriminator_specific.compile(loss='mse', optimizer='adam', metrics=['mse'])

        # build and compile the domain-shared discriminator
        self.discriminator_shared = self.build_discriminator(name = 'shared_discriminator')
        self.discriminator_shared.compile(loss='mse', optimizer='adam', metrics=['mse'])

        # build the generator
        self.generator = self.build_generator()

        # generator takes the multimodal input as the input
        z = Input(shape=(self.input_shape,))
        # generator returns the fake news label (main_pred), reconstructed input (aux_pred), latent embeddings (latent_emb)
        main_pred, aux_pred, latent_emb = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator_specific.trainable = False
        self.discriminator_shared.trainable = False

        # The discriminator takes generated latent spaces as inputs
        domain_pred_specific = self.discriminator_specific(latent_emb[:,:int(self.latent_dim/2)])
        domain_pred_shared = self.discriminator_shared(latent_emb[:,int(self.latent_dim/2):])

        # The combined model
        self.combined = Model(z, [main_pred, aux_pred, domain_pred_specific, domain_pred_shared])
        self.combined.compile(loss= ['binary_crossentropy', 'mse', 'mse', 'mse'],
                loss_weights=[1, self.lambda1, self.lambda2, self.lambda3],
                metrics=['accuracy'], optimizer='adam')

    def build_generator(self):
        input_embedding = Input(shape = (self.input_shape,), name = 'input_embedding')
        input_hidden_embedding = Dense(int(self.latent_dim/2), activation='relu')(input_embedding)
        latent_layer = Dense(self.latent_dim, activation='relu', name = 'emb_layer')(input_hidden_embedding)
        output_classifier = Dense(1, activation='sigmoid', name = 'output_classifier')(latent_layer)
        output_hidden_decoder = Dense(int(self.latent_dim/2), activation='relu')(latent_layer)
        output_decoder = Dense(self.input_shape, name = 'output_decoder')(output_hidden_decoder)
        model = Model(input_embedding, [output_classifier, output_decoder, latent_layer], name = 'generator')
        return model

    def build_discriminator(self, name = 'discriminator'):
        latent_input = Input(shape=(int(self.latent_dim/2),))
        domain_hidden_layer = Dense((self.no_domains*2), activation='sigmoid')(latent_input)
        output_domain_classifier = Dense(self.no_domains, activation='sigmoid', name = 'output_domain_classifier')(domain_hidden_layer)
        model =  Model(latent_input, output_domain_classifier, name = name)
        return model

    def train(self, X_train, X_test, y_train, y_test, yd_train, yd_test, epochs, batch_size):
        test_accuracies = []
        train_accuracies = []
        for epoch in range(epochs):
            no_batches = int(X_train.shape[0]/batch_size)
            for batch_id in range(no_batches):
                # select batches randomly
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                X_b = X_train[idx]
                y_b = y_train[idx]
                yd_b = yd_train[idx]

                # train discimninators using the current latent spaces
                main_out, aux_out, latent_layer = self.generator.predict(X_b)
                d1_loss = self.discriminator_specific.train_on_batch(latent_layer[:, :int(self.latent_dim/2)], yd_b)
                d2_loss = self.discriminator_shared.train_on_batch(latent_layer[:, int(self.latent_dim/2):], yd_b)

                # Train the generator
                g_loss = self.combined.train_on_batch(X_b, [y_b, X_b, yd_b, yd_b])
            results = self.combined.evaluate(X_train, [y_train, X_train, yd_train, yd_train], verbose=0)
            train_accuracies.append(results[5])
            print('loss: {} accuracy: {}'.format(results[0], results[5]))
            results = self.combined.evaluate(X_test, [y_test, X_test, yd_test, yd_test], verbose=0)
            test_accuracies.append(results[5])

    def evaluate(self, X_test, y_test, yd_test):
        results = self.combined.evaluate(X_test, [y_test, X_test, yd_test, yd_test], verbose=0)
        y_test_pred, _, _, _ = self.combined.predict(X_test)
        y_test_pred[y_test_pred>0.5] = 1
        y_test_pred[y_test_pred<=0.5] = 0
        p, r, f , _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro')
        print('accuracy: {} precision: {} recall: {} f-score: {}'.format(results[5], p, r, f))
        return results[5], p, r, f,

    def get_latent_space(self, X_test):
        _, _, latent_embs = self.generator.predict(X_test)
        return latent_embs

if __name__== '__main__':
    input_d, domain_emb_d, latent_d, lambda1, lambda2, lambda3 = 1024, 100, 512, 1, 5, 10
    model = FAKE_NEWS_DETECTOR(input_d, domain_emb_d, latent_d, lambda1, lambda2, lambda3)
