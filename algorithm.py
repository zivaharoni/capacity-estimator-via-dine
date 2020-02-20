# from builtins import *
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('axes', labelsize=14)
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.backend.set_floatx('float32')

from tensorflow.python.keras import backend as K
from rnn_modified import LSTMNew


class Algorithm(object):
    def __init__(self, config):
        self.config = config
        self.directory = config.directory
        self.channel_name = config.config_name


        self.feedback = config.feedback
        self.P = config.P
        self.n = config.n
        self.m = config.m
        self.T = config.T
        self.batch_size = config.batch_size
        self.epochs_di = config.epochs_di
        self.epochs_enc = config.epochs_enc
        self.n_steps_mc = config.n_steps_mc
        self.noise_std = np.sqrt(config.N)
        self.capacity = config.C

        self.DI_hidden = config.DI_hidden
        self.DI_last_hidden = config.DI_last_hidden
        self.DI_dropout = config.DI_dropout
        self.NDT_hidden = config.NDT_hidden
        self.NDT_last_hidden = config.NDT_last_hidden
        self.NDT_dropout = config.NDT_dropout
        self.stateful = True

        if config.opt == "sgd":
            self._opt = keras.optimizers.SGD
            self._opt_kwargs = {'momentum': 0.99}
        elif config.opt == "adam":
            self._opt = keras.optimizers.Adam
            self._opt_kwargs = {}
        else:
            raise ValueError("invalid optimizer was chosen")


        self.clip_norm = config.clip_norm
        self.lr_DI = config.lr_rate_DI
        self.lr_enc = config.lr_rate_enc


        self.h_y_model, self.h_xy_model, self.DI_model = self._build_DI_model()
        self.channel = self._build_channel()
        self.encoder = self._build_encoder()

        self.mean_T_y, self.mean_T_y_tild, self.mean_T_xy, self.mean_T_xy_tild = \
            (keras.metrics.Mean(), keras.metrics.Mean(), keras.metrics.Mean(), keras.metrics.Mean())

    def _build_DI_model(self):
        def build_DV(name, input_shape):
            randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
            bias_init = keras.initializers.Constant(0.01)

            lstm = LSTMNew(self.DI_hidden, return_sequences=True, name=name, stateful=self.stateful,
                           dropout=self.DI_dropout, recurrent_dropout=self.DI_dropout)
            split = layers.Lambda(tf.split, arguments={'axis': -1, 'num_or_size_splits': 2})
            squeeze = layers.Lambda(tf.squeeze, arguments={'axis': -1})
            dense0 = layers.Dense(self.DI_hidden, bias_initializer=bias_init, kernel_initializer=randN_05, activation="elu")
            dense1 = layers.Dense(self.DI_last_hidden, bias_initializer=bias_init, kernel_initializer=randN_05, activation="elu")
            dense2 = layers.Dense(1, bias_initializer=bias_init, kernel_initializer=randN_05, activation=None)

            in_ = layers.Input(batch_shape=input_shape)
            t = lstm(in_)
            t_1, t_2 = split(t)
            t_1, t_2 = squeeze(t_1), squeeze(t_2)
            t_1, t_2 = dense0(t_1), dense0(t_2)
            t_1, t_2 = dense1(t_1), dense1(t_2)
            t_1, t_2 = dense2(t_1), K.exp(dense2(t_2))
            model = keras.models.Model(inputs=in_, outputs=[t_1, t_2])
            return model


        h_y_model  = build_DV("LSTM_y",  [self.batch_size, self.T, 2 * self.n])
        h_xy_model = build_DV("LSTM_xy", [self.batch_size, self.T, 4 * self.n])

        DI_model = keras.models.Model(inputs=h_y_model.inputs + h_xy_model.inputs,
                                      outputs=h_y_model.outputs + h_xy_model.outputs)
        return h_y_model, h_xy_model, DI_model

    def _build_channel(self):
        if self.channel_name == "awgn":
            channel = AWGN(self.noise_std, [self.batch_size, 1, self.n])
        elif self.channel_name in ["arma_ff", "arma_fb"]:
            channel = ARMA_AWGN(self.config.channel_alpha, self.noise_std, [self.batch_size, 1, self.n])
        else:
            raise ValueError("Invalid channel name")

        return channel

    def _build_encoder(self):
        def forward():
            encoder_transform = keras.models.Sequential([
                keras.layers.LSTM(self.NDT_hidden, return_sequences=True, name="LSTM_enc", stateful=self.stateful,
                                  batch_input_shape=[self.batch_size, 1, self.n+self.m],
                                  dropout=self.NDT_dropout, recurrent_dropout=self.NDT_dropout),
                keras.layers.Dense(self.NDT_hidden, activation="elu"),
                keras.layers.Dense(self.NDT_last_hidden, activation="elu"),
                keras.layers.Dense(self.n, activation=None),
                norm_layer])

            enc_out = list()
            channel_out = list()
            enc_in_stable = keras.layers.Input(shape=[self.T, self.m])
            enc_split = tf.split(enc_in_stable, num_or_size_splits=self.T, axis=1)
            enc_in_feedback = keras.layers.Input(shape=[1, self.n])

            enc_in_0 = tf.concat([enc_split[0], enc_in_feedback], axis=-1)
            for t in range(self.T):
                if t == 0:
                    enc_out.append(encoder_transform(enc_in_0))
                else:
                    enc_in_t = tf.concat([enc_split[t], enc_out[t - 1]], axis=-1)
                    enc_out.append(norm_layer(encoder_transform(enc_in_t)))
                channel_out.append(channel(enc_out[t]))

            channel_out = tf.concat(channel_out, axis=1)
            enc_out = tf.concat(enc_out, axis=1)

            encoder = keras.models.Model(inputs=[enc_in_stable, enc_in_feedback], outputs=[enc_out, channel_out])

            return encoder

        def feedback():
            encoder_transform = keras.models.Sequential([
                keras.layers.LSTM(self.NDT_hidden, return_sequences=True, name="LSTM_enc", stateful=self.stateful,
                                  batch_input_shape=[self.batch_size, 1,2*self.n+self.m],
                                  recurrent_dropout=self.NDT_dropout, dropout=self.NDT_dropout),
                keras.layers.Dense(self.NDT_hidden, activation="elu"),
                keras.layers.Dense(self.NDT_last_hidden, activation="elu"),
                keras.layers.Dense(self.n, activation=None),
                norm_layer])

            enc_out = list()
            channel_out = list()
            enc_in_stable = keras.layers.Input(shape=[self.T, self.m])
            enc_split = tf.split(enc_in_stable, num_or_size_splits=self.T, axis=1)
            enc_in_feedback = keras.layers.Input(shape=[1, 2*self.n])

            enc_in_0 = tf.concat([enc_split[0], enc_in_feedback], axis=-1)
            for t in range(self.T):
                if t == 0:
                    enc_out.append(encoder_transform(enc_in_0))
                else:
                    enc_in_t = tf.concat([enc_split[t], enc_out[t-1], channel_out[t - 1]], axis=-1)
                    enc_out.append(norm_layer(encoder_transform(enc_in_t)))
                channel_out.append(channel(enc_out[t]))

            channel_out = tf.concat(channel_out, axis=1)
            enc_out = tf.concat(enc_out, axis=1)

            encoder = keras.models.Model(inputs=[enc_in_stable, enc_in_feedback], outputs=[enc_out, channel_out])

            return encoder

        norm_layer = keras.layers.Lambda(lambda x: tf.divide(x, tf.sqrt(tf.reduce_mean(tf.square(x)))) * tf.sqrt(self.P))

        channel_layer = keras.layers.Lambda(lambda x: x + self.channel.call())
        channel = keras.models.Sequential([channel_layer])

        encoder = feedback() if self.feedback else forward()


        return encoder

    def random_sample(self):
        msg = tf.random.normal(shape=[self.batch_size, self.T, self.m], dtype=tf.float32)*tf.sqrt(self.P)
        return msg

    def rand_enc_samples(self, amount):

        if self.feedback:
            S = tf.stack([self.encoder([self.random_sample(), np.zeros([self.batch_size,1, 2*self.n])], training=False)[0] for _ in range(amount)], axis=-1)
        else:
            # S = tf.stack([self.encoder(self.random_sample(), training=False)[0] for _ in range(amount)], axis=-1)
            S = tf.stack([self.encoder([self.random_sample(), np.zeros([self.batch_size,1, self.n])], training=False)[0] for _ in range(amount)], axis=-1)

        return S

    @staticmethod
    def DV_loss(t_list):
        t_1, t_2 = t_list
        return K.mean(t_1) - K.log(K.mean(t_2))

    @staticmethod
    def DI_data(x, y):
        y_tilde = tf.random.normal(tf.shape(y))
        input_A = tf.concat([y, y_tilde], axis=-1)
        input_B1 = tf.concat([x, y], axis=-1)
        input_B2 = tf.concat([x, y_tilde], axis=-1)
        input_B = tf.concat([input_B1, input_B2], axis=-1)
        return input_A, input_B

    def save(self):

        self.h_y_model.save(os.path.join(self.directory,'h_y_model.h5'))
        self.h_xy_model.save(os.path.join(self.directory,'h_xy_model.h5'))
        self.encoder.save(os.path.join(self.directory,'di_model.h5'))

    def hist_X(self, X, title, save=False):
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(X, bins=100)
        plt.title(title + " sample size: {:d}, mean: {:2.3f} std: {:2.3f}".format(X.shape[0], np.mean(X), np.std(X)))
        plt.xlabel('alphabet')
        plt.ylabel('density')
        if save:
            fig_file_name = '{}.png'.format("-".join(title.lower().split(" ")))
            plt.savefig(os.path.join(self.directory,fig_file_name))
        plt.close()

    def plot(self, data, title, save=False):
        data = np.array(data)
        # ma_window = np.minimum(50, data.shape[0])
        # plt.plot(np.convolve(data, np.ones(ma_window)/ma_window, mode='same'), label='model')
        plt.plot(data, label='model')
        if self.capacity is not None:
            plt.plot(np.ones_like(data) * self.capacity, label='ground truth')
        plt.legend()
        plt.xlabel('#of updates')
        plt.ylabel('Directed Info.')
        plt.ylim(np.percentile(data,1), np.maximum(np.percentile(data,99),self.capacity)*1.1)
        plt.title(title)
        plt.grid(True)
        if save:
            fig_file_name = '{}.png'.format("-".join(title.lower().split(" ")))
            plt.savefig(os.path.join(self.directory,fig_file_name))
        plt.close()

    def evaluate_encoder(self, n_steps=5, bar=True):
        self.mean_T_y.reset_states()
        self.mean_T_xy.reset_states()
        self.mean_T_y_tild.reset_states()
        self.mean_T_xy_tild.reset_states()

        if bar:
            rang = tqdm(range(1, n_steps + 1))
        else:
            rang = range(1, n_steps + 1)

        self.channel.reset_states()
        self.encoder.reset_states()
        self.h_y_model.reset_states()
        self.h_xy_model.reset_states()

        if self.feedback:
            y_feedback = tf.convert_to_tensor(np.zeros([self.batch_size, 1, 2*self.n]))
        else:
            y_feedback = tf.convert_to_tensor(np.zeros([self.batch_size, 1, self.n]))

        for iter in rang:
            X_batch = self.random_sample()
            if self.feedback:
                X_batch = [X_batch, y_feedback]
            else:
                X_batch = [X_batch, y_feedback]
            x_enc, y_recv = self.encoder(X_batch, training=False)
            if self.feedback:
                y_feedback = tf.concat([tf.reshape(x_enc[:, -1, :], [self.batch_size, 1, self.n]),
                                        tf.reshape(y_recv[:, -1, :], [self.batch_size, 1, self.n])],
                                       axis=-1)
                # y_feedback = tf.reshape(y_recv[:, -1, :], [self.batch_size, 1, self.n])
            else:
                y_feedback = tf.expand_dims(x_enc[:, -1, :], axis=1)
            joint_marg_s = self.DI_data(x_enc, y_recv)

            T_y = self.h_y_model(joint_marg_s[0], training=False)
            T_xy = self.h_xy_model(joint_marg_s[1], training=False)

            if iter >= np.minimum(100, n_steps//10):
                mi_y_avg = (self.mean_T_y(T_y[0]), self.mean_T_y_tild(T_y[1]))
                mi_xy_avg = (self.mean_T_xy(T_xy[0]), self.mean_T_xy_tild(T_xy[1]))

        mi_avg = self.DV_loss(mi_xy_avg) - self.DV_loss(mi_y_avg)

        return mi_avg

    def train_mi(self, n_epochs=5):
        optimizer_y = self._opt(learning_rate=self.lr_DI, **self._opt_kwargs)
        optimizer_xy = self._opt(learning_rate=self.lr_DI, **self._opt_kwargs)

        history_mi = list()
        self.channel.reset_states()
        self.encoder.reset_states()
        self.h_y_model.reset_states()
        self.h_xy_model.reset_states()

        if self.feedback:
            y_feedback = tf.convert_to_tensor(np.zeros([self.batch_size, 1, 2*self.n]))
        else:
            y_feedback = tf.convert_to_tensor(np.zeros([self.batch_size, 1, self.n]))

        for epoch in tqdm(range(1, n_epochs + 1)):
            if epoch % (n_epochs // 5) == 0 and epoch > 0:
                self.plot(history_mi, 'Training Proces', save=True)

            X_batch = self.random_sample()
            if self.feedback:
                X_batch = [X_batch, y_feedback]
            else:
                X_batch = [X_batch, y_feedback]
            x_enc, y_recv = self.encoder(X_batch, training=True)
            if self.feedback:
                y_feedback = tf.concat([tf.reshape(x_enc[:, -1, :], [self.batch_size, 1, self.n]),
                                        tf.reshape(y_recv[:, -1, :], [self.batch_size, 1, self.n])],
                                       axis=-1)
                # y_feedback = tf.reshape(y_recv[:, -1, :], [self.batch_size, 1, self.n])
            else:
                y_feedback = tf.expand_dims(x_enc[:, -1, :], axis=1)

            joint_marg_s = self.DI_data(x_enc, y_recv)

            with tf.GradientTape(persistent=True) as tape:
                T_y = self.h_y_model(joint_marg_s[0], training=True)
                loss_y = -self.DV_loss(T_y)
                T_xy = self.h_xy_model(joint_marg_s[1], training=True)
                loss_xy = -self.DV_loss(T_xy)
                loss = loss_xy - loss_y

            gradients = tape.gradient(loss_y, self.h_y_model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
            optimizer_y.apply_gradients(zip(gradients, self.h_y_model.trainable_variables))
            gradients = tape.gradient(loss_xy, self.h_xy_model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
            optimizer_xy.apply_gradients(zip(gradients, self.h_xy_model.trainable_variables))

            history_mi.append(-loss)

            # if epoch == n_epochs//10:
            #     optimizer_y = keras.optimizers.SGD(lr=0.01, clipnorm=self.clipnorm)
            #     optimizer_xy = keras.optimizers.SGD(lr=0.01, clipnorm=self.clipnorm)

        return history_mi

    def train_encoder(self, n_epochs=5):

        optimizer_y = self._opt(learning_rate=self.lr_DI, **self._opt_kwargs)
        optimizer_xy = self._opt(learning_rate=self.lr_DI, **self._opt_kwargs)
        optimizer_ae = self._opt(learning_rate=self.lr_enc, **self._opt_kwargs)



        history_mi = list()
        self.channel.reset_states()
        self.encoder.reset_states()
        self.h_y_model.reset_states()
        self.h_xy_model.reset_states()

        if self.feedback:
            y_feedback = tf.convert_to_tensor(np.zeros([self.batch_size, 1, 2*self.n]))
        else:
            y_feedback = tf.convert_to_tensor(np.zeros([self.batch_size, 1, self.n]))

        GRADS = list()
        for epoch in tqdm(range(1, n_epochs + 1)):
            if epoch > 100:#% (n_epochs // 100) == 0 and epoch > 0:
                self.plot(history_mi, 'Training Encoder Process', save=True)

            X_batch = self.random_sample()
            with tf.GradientTape(persistent=True) as tape:
                if self.feedback:
                    X_batch = [X_batch, y_feedback]
                else:
                    X_batch = [X_batch, y_feedback]
                x_enc, y_recv = self.encoder(X_batch, training=True)
                if self.feedback:
                    y_feedback = tf.concat([tf.reshape(x_enc[:, -1, :], [self.batch_size, 1, self.n]),
                                            tf.reshape(y_recv[:, -1, :], [self.batch_size, 1, self.n])],
                                           axis=-1)
                    # y_feedback = tf.reshape(y_recv[:, -1, :], [self.batch_size, 1, self.n])
                else:
                    y_feedback = tf.expand_dims(x_enc[:, -1, :], axis=1)
                joint_marg_s = self.DI_data(x_enc, y_recv)
                T_y = self.h_y_model(joint_marg_s[0], training=True)
                loss_y = -self.DV_loss(T_y)
                T_xy = self.h_xy_model(joint_marg_s[1], training=True)
                loss_xy = -self.DV_loss(T_xy)

            gradients = tape.gradient(loss_y, self.h_y_model.trainable_variables)
            GRADS.append(gradients)
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
            optimizer_y.apply_gradients(zip(gradients, self.h_y_model.trainable_variables))
            gradients = tape.gradient(loss_xy, self.h_xy_model.trainable_variables)
            GRADS.append(gradients)
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
            optimizer_xy.apply_gradients(zip(gradients, self.h_xy_model.trainable_variables))

            X_batch = self.random_sample()
            with tf.GradientTape() as tape:
                if self.feedback:
                    X_batch = [X_batch, y_feedback]
                else:
                    X_batch = [X_batch, y_feedback]
                x_enc, y_recv = self.encoder(X_batch, training=True)
                if self.feedback:
                    y_feedback = tf.concat([tf.reshape(x_enc[:, -1, :], [self.batch_size, 1, self.n]),
                                            tf.reshape(y_recv[:, -1, :], [self.batch_size, 1, self.n])],
                                           axis=-1)
                    # y_feedback = tf.reshape(y_recv[:, -1, :], [self.batch_size, 1, self.n])
                else:
                    y_feedback = tf.expand_dims(x_enc[:, -1, :], axis=1)

                joint_marg_s = self.DI_data(x_enc, y_recv)
                T_y = self.h_y_model(joint_marg_s[0], training=True)
                loss_y = -self.DV_loss(T_y)
                T_xy = self.h_xy_model(joint_marg_s[1], training=True)
                loss_xy = -self.DV_loss(T_xy)

                loss = loss_xy - loss_y

            gradients = tape.gradient(loss, self.encoder.trainable_variables)
            GRADS.append(gradients)
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
            optimizer_ae.apply_gradients(zip(gradients, self.encoder.trainable_variables))

            history_mi.append(-loss)
            GRADS.append([-loss])

            print(*('{:5.2f}'.format(float(tf.linalg.global_norm(g))) for g in GRADS))
            GRADS = list()

        return history_mi


class ARMA_AWGN(object):
    def __init__(self, alpha, std, shape):
        self.shape = shape
        self.alpha = alpha
        self.last_n = np.zeros(shape)
        self.std = std

    def call(self):
        new_n = np.random.randn(*self.shape) * self.std
        z = self.alpha * self.last_n + new_n
        self.last_n = new_n
        return z

    def reset_states(self):
        self.last_n = np.zeros(self.shape)


class AWGN(object):
    def __init__(self, std, shape):
        self.shape = shape
        self.std = std

    def call(self):
        z = np.random.randn(*self.shape) * self.std
        return z

    def reset_states(self):
        pass

