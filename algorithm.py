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
    def __init__(self,n, P, channel_configs, m, T, batch, epochs_train_di, epochs_train_enc, clipnorm, n_steps_mc, stateful, directory, feedback=False):
        self.directory = directory

        self.feedback = feedback
        self.P = P
        self.n = n              # dimension of the R.V. X
        self.m = m              # rand generator dimension at encoder
        self.T = T              # unroll sequential model T time-steps
        self.batch_size = batch
        self.epochs_di = epochs_train_di
        self.epochs_enc = epochs_train_enc
        self.n_steps_mc = n_steps_mc
        self.stateful = stateful
        self.noise_std = np.sqrt(channel_configs["noise_var"])
        self.capacity = channel_configs["capacity"]

        self._opt = keras.optimizers.Adam
        self.clipnorm = clipnorm#1.0#0.25
        self.lr_schedule = 0.0001
        # self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.0005,
        #     decay_steps=100,
        #     decay_rate=0.96,
        #     staircase=True)

        self.h_y_model, self.h_xy_model, self.DI_model = self._build_DI_model()
        self.channel_configs = channel_configs
        self.channel = self._build_channel()
        self.encoder = self._build_encoder()

        self.mean_T_y, self.mean_T_y_tild, self.mean_T_xy, self.mean_T_xy_tild = \
            (keras.metrics.Mean(), keras.metrics.Mean(), keras.metrics.Mean(), keras.metrics.Mean())

    def _build_DI_model(self):
        def build_DV(name, input_shape):
            randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
            bias_init = keras.initializers.Constant(0.01)

            lstm = LSTMNew(500, return_sequences=True, name=name, stateful=self.stateful)#, dropout=0.5, recurrent_dropout=0.5)
            split = layers.Lambda(tf.split, arguments={'axis': -1, 'num_or_size_splits': 2})
            squeeze = layers.Lambda(tf.squeeze, arguments={'axis': -1})
            dense0 = layers.Dense(500, bias_initializer=bias_init, kernel_initializer=randN_05, activation="elu")
            dense1 = layers.Dense(256, bias_initializer=bias_init, kernel_initializer=randN_05, activation="elu")
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
        channel_configs = self.channel_configs

        if channel_configs["name"] == "awgn":
            channel = AWGN(self.noise_std, [self.batch_size, 1, self.n])
        elif channel_configs["name"] == "arma_awgn":
            channel = ARMA_AWGN(channel_configs["alpha"], self.noise_std, [self.batch_size, 1, self.n])
        else:
            raise ValueError("Invalid channel name")

        return channel

    def _build_encoder(self):
        def forward():
            # encoder_only = keras.models.Sequential([
            #     keras.layers.LSTM(500, return_sequences=True, name="LSTM_enc", stateful=self.stateful,
            #                       batch_input_shape=[self.batch_size, self.T, self.m], recurrent_dropout=0.5, dropout=0.5),
            #     keras.layers.Dense(500, activation="elu"),
            #     keras.layers.Dense(100, activation="elu"),
            #     keras.layers.Dense(self.n, activation=None),
            #     norm_layer])
            #
            # enc_out_split = tf.split(encoder_only.output, num_or_size_splits=self.T, axis=1)
            # channel_out = list()
            # for t in range(self.T):
            #     channel_out.append(channel(enc_out_split[t]))
            # channel_out = tf.concat(channel_out, axis=1)
            # encoder = keras.models.Model(inputs=encoder_only.input, outputs=[encoder_only.output, channel_out])
            encoder_transform = keras.models.Sequential([
                keras.layers.LSTM(500, return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[self.batch_size, 1, self.n+self.m]),#, dropout=0.5, recurrent_dropout=0.5),
                keras.layers.Dense(500, activation="elu"),
                keras.layers.Dense(100, activation="elu"),
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
                keras.layers.LSTM(500, return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[self.batch_size, 1,2*self.n+self.m]),#, recurrent_dropout=0.75, dropout=0.75),
                keras.layers.Dense(500, activation="elu"),
                keras.layers.Dense(100, activation="elu"),
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
        y_tilde = tf.random.uniform(tf.shape(y), minval=K.min(y), maxval=K.max(y))
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
        plt.plot(np.ones_like(data) * self.capacity, label='ground truth')
        plt.legend()
        plt.xlabel('#of updates')
        plt.ylabel('Directed Info.')
        # plt.ylim(np.minimum(np.min(data), -0.05), self.capacity * 1.5)
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
        optimizer_y = self._opt(lr=self.lr_schedule, clipnorm=self.clipnorm)
        optimizer_xy = self._opt(lr=self.lr_schedule, clipnorm=self.clipnorm)

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
            optimizer_y.apply_gradients(zip(gradients, self.h_y_model.trainable_variables))
            gradients = tape.gradient(loss_xy, self.h_xy_model.trainable_variables)
            optimizer_xy.apply_gradients(zip(gradients, self.h_xy_model.trainable_variables))

            history_mi.append(-loss)

            # if epoch == n_epochs//10:
            #     optimizer_y = keras.optimizers.SGD(lr=0.01, clipnorm=self.clipnorm)
            #     optimizer_xy = keras.optimizers.SGD(lr=0.01, clipnorm=self.clipnorm)

        return history_mi

    def train_encoder(self, n_epochs=5):

        optimizer_y = self._opt(lr=self.lr_schedule, clipnorm=self.clipnorm)
        optimizer_xy = self._opt(lr=self.lr_schedule, clipnorm=self.clipnorm)
        optimizer_ae = self._opt(lr=self.lr_schedule/2, clipnorm=self.clipnorm)



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
            optimizer_y.apply_gradients(zip(gradients, self.h_y_model.trainable_variables))
            gradients = tape.gradient(loss_xy, self.h_xy_model.trainable_variables)
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
            optimizer_ae.apply_gradients(zip(gradients, self.encoder.trainable_variables))

            history_mi.append(-loss)
            # if epoch == n_epochs//10:
            #     optimizer_y = keras.optimizers.SGD(lr=0.01, clipnorm=self.clipnorm)
            #     optimizer_xy = keras.optimizers.SGD(lr=0.01, clipnorm=self.clipnorm)
            #     optimizer_ae = keras.optimizers.SGD(lr=0.01, clipnorm=self.clipnorm)

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

