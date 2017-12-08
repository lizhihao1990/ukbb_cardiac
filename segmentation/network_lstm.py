# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from network import *


def UNet(images, n_class, n_level, n_filter, n_block, training):
    """
        U-Net for segmenting an input image into n_class classes.
        """
    net = {}
    x = images

    # Downsampling or analysis path
    # Learn fine-to-coarse features at each resolution level
    for l in range(0, n_level):
        with tf.name_scope('conv{}'.format(l)):
            # If this is the first level (l = 0), keep the resolution.
            # Otherwise, convolve with a stride of 2, i.e. downsample by a
            # factor of 2.
            strides = 1 if l == 0 else 2
            # For each resolution level, perform n_block[l] times convolutions
            x = conv2d_bn_relu(x, filters=n_filter[l], training=training,
                               kernel_size=3, strides=strides)
            for i in range(1, n_block[l]):
                x = conv2d_bn_relu(x, filters=n_filter[l], training=training,
                                   kernel_size=3)
            net['conv{}'.format(l)] = x

    # Upsampling or synthesis path
    net['conv{}_up'.format(n_level - 1)] = net['conv{}'.format(n_level - 1)]
    for l in range(n_level - 2, -1, -1):
        with tf.name_scope('conv{}_up'.format(l)):
            x = conv2d_transpose_bn_relu(net['conv{}_up'.format(l + 1)],
                                         filters=n_filter[l],
                                         training=training,
                                         kernel_size=3, strides=2)
            x = tf.concat([net['conv{}'.format(l)], x], axis=-1)
            for i in range(0, n_block[l]):
                x = conv2d_bn_relu(x, filters=n_filter[l], training=training,
                                   kernel_size=3)
            net['conv{}_up'.format(l)] = x

    # Perform prediction
    with tf.name_scope('out'):
        # We only calculate logits, instead of softmax here because the loss
        # function tf.nn.softmax_cross_entropy() accepts the unscaled logits
        # and performs softmax internally for efficiency and numerical
        # stability reasons.
        # Refer to https://github.com/tensorflow/tensorflow/issues/2462
        logits = tf.layers.conv2d(net['conv0_up'], filters=n_class,
                                  kernel_size=1, padding='same')
    return logits, net


def Conv_LSTM(features, n_hidden):
    """
        Convolutional LSTM which processes a feature map.

        features: NTXYC
        """
    features_shape = tf.shape(features)
    cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=features_shape[2:],
                                         output_channels=n_hidden,
                                         kernel_shape=(3, 3))

    initial_state = cell.zero_state(features_shape[0], tf.float32)
    state = initial_state
    # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    outputs = []
    num_steps = features_shape[1]

    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            cell_output, state = cell(features[:, time_step], state)
            outputs.append(cell_output)
    return outputs


def UNet_Model(images, labels, n_class, n_level, n_filter, n_block, training):
    """
        A model which takes input images, builds the graph and outputs the loss.

        images: NXYC
        labels: NXY
        """
    logits, net = UNet(images=images, n_class=n_class, n_level=n_level,
                       n_filter=n_filter, n_block=n_block, training=training)

    label_1hot = tf.one_hot(indices=labels, depth=n_class)
    label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_1hot,
                                                         logits=logits)
    loss = tf.reduce_mean(label_loss)
    return loss


def UNet_LSTM_Model(images, labels, n_class, n_level, n_filter, n_block, training):
    """
        A model which takes input images, builds the graph and outputs the loss.

        images: NTXYC
        labels: NTXY
        """
    # Merge the temporal dimension into the batch dimension
    images_shape = tf.shape(images)
    images = tf.reshape(images, (-1, images_shape[2], images.shape[3], images.shape[4]))

    # Generate the feature map using the UNet
    # images: (N*T)XYC
    _, net = UNet(images=images, n_class=n_class, n_level=n_level,
                  n_filter=n_filter, n_block=n_block, training=training)

    # features: (N*T)XYC
    features = net['conv0_up']
    features_shape = tf.shape(features)
    # features: NTXYC
    features = tf.reshape(features, (images_shape[0], images_shape[1],
                                     images_shape[2], images_shape[3], -1))

    # Pass the feature map to the LSTM
    # outputs: TNXYC
    outputs = Conv_LSTM(features, n_hidden=16)

    # For the moment, only focus on the last time frame, where we have annotations
    # So this is the loss for the last time frame
    # TODO: sequence to sequence learning
    # logis: NXYC
    logits = tf.layers.conv2d(outputs[-1], filters=n_class, kernel_size=1,
                              padding='same')

    label_1hot = tf.one_hot(indices=labels[:, -1], depth=n_class)
    label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_1hot,
                                                         logits=logits)
    loss = tf.reduce_mean(label_loss)
    return loss
