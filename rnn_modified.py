import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import RNN
from tensorflow.keras.layers import LSTM, LSTMCell
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import initializers
from tensorflow.python.keras import activations
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.training.tracking import data_structures




class LSTMCellNew(LSTMCell):
    """
    Modified implementation of the LSTM cell such that It has 2 inputs: (y, y_bar).
    The recurrent state (h) is calculated for each input, but the input state for both states is only h(y)
    output is a pair of states sequences (h(y),h_bar(y))
    """

    # Create a new build function to suite our needs:
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(LSTMCellNew, self).__init__(units, **kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        # tuple(_ListWrapper) was silently dropping list content in at least 2.7.10,
        # and fixed after 2.7.16. Converting the state_size to wrapper around
        # NoDependency(), so that the base_layer.__setattr__ will not convert it to
        # ListWrapper. Down the stream, self.states will be a list since it is
        # generated from nest.map_structure with list, and tuple(list) will work
        # properly.
        self.state_size = data_structures.NoDependency([self.units, self.units])
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim // 2, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([self.bias_initializer((self.units,), *args, **kwargs),
                                          initializers.Ones()((self.units,), *args, **kwargs),
                                          self.bias_initializer((self.units * 2,), *args, **kwargs),
                                          ])
            else:
                bias_initializer = self.bias_initializer

            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
        f = self.recurrent_activation(x_f + K.dot(
            h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(
            h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def call(self, inputs_stacked, states, training=None):
        def cell_(inputs, h_tm1, c_tm1):
            dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
            rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
                h_tm1, training, count=4)

            if self.implementation == 1:
                if 0 < self.dropout < 1.:
                    inputs_i = inputs * dp_mask[0]
                    inputs_f = inputs * dp_mask[1]
                    inputs_c = inputs * dp_mask[2]
                    inputs_o = inputs * dp_mask[3]
                else:
                    inputs_i = inputs
                    inputs_f = inputs
                    inputs_c = inputs
                    inputs_o = inputs
                k_i, k_f, k_c, k_o = array_ops.split(self.kernel, num_or_size_splits=4, axis=1)
                x_i = K.dot(inputs_i, k_i)
                x_f = K.dot(inputs_f, k_f)
                x_c = K.dot(inputs_c, k_c)
                x_o = K.dot(inputs_o, k_o)
                if self.use_bias:
                    b_i, b_f, b_c, b_o = array_ops.split(self.bias, num_or_size_splits=4, axis=0)
                    x_i = K.bias_add(x_i, b_i)
                    x_f = K.bias_add(x_f, b_f)
                    x_c = K.bias_add(x_c, b_c)
                    x_o = K.bias_add(x_o, b_o)

                if 0 < self.recurrent_dropout < 1.:
                    h_tm1_i = h_tm1 * rec_dp_mask[0]
                    h_tm1_f = h_tm1 * rec_dp_mask[1]
                    h_tm1_c = h_tm1 * rec_dp_mask[2]
                    h_tm1_o = h_tm1 * rec_dp_mask[3]
                else:
                    h_tm1_i = h_tm1
                    h_tm1_f = h_tm1
                    h_tm1_c = h_tm1
                    h_tm1_o = h_tm1
                x = (x_i, x_f, x_c, x_o)
                h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
                c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
            else:
                if 0. < self.dropout < 1.:
                    inputs = inputs * dp_mask[0]
                z = K.dot(inputs, self.kernel)
                if 0. < self.recurrent_dropout < 1.:
                    h_tm1 = h_tm1 * rec_dp_mask[0]
                z += K.dot(h_tm1, self.recurrent_kernel)
                if self.use_bias:
                    z = K.bias_add(z, self.bias)

                z = array_ops.split(z, num_or_size_splits=4, axis=1)
                c, o = self._compute_carry_and_output_fused(z, c_tm1)

            h = o * self.activation(c)
            return h, c

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        in_1, in_2 = tf.split(inputs_stacked, num_or_size_splits=2, axis=-1)

        h_1, c_1 = cell_(in_1, h_tm1, c_tm1)
        h_2, _ = cell_(in_2, h_tm1, c_tm1)

        h = K.stack([h_1, h_2], axis=-1)
        recurrent_states = [h_1, c_1]
        return h, recurrent_states


class LSTMNew(RNN):
    # @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=False,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=2,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        cell = LSTMCellNew(units,
                           activation=activation,
                           recurrent_activation=recurrent_activation,
                           use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           recurrent_initializer=recurrent_initializer,
                           unit_forget_bias=unit_forget_bias,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           recurrent_regularizer=recurrent_regularizer,
                           bias_regularizer=bias_regularizer,
                           kernel_constraint=kernel_constraint,
                           recurrent_constraint=recurrent_constraint,
                           bias_constraint=bias_constraint,
                           dropout=dropout,
                           recurrent_dropout=recurrent_dropout,
                           implementation=implementation)
        super(LSTMNew, self).__init__(cell,
                                      return_sequences=return_sequences,
                                      return_state=return_state,
                                      go_backwards=go_backwards,
                                      stateful=stateful,
                                      unroll=unroll,
                                      **kwargs)
