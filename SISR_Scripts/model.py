import tensorflow as tf


# each DB consists of 8 convolutional layers, each producing 16 feature maps
# so each DB produces 128 feature maps total
# layers receive output of all preceding layers as input 
# Inputs: 
#       x: the output from the previous block for skip connections
#       num_layers: number of convolution layers in the dense block
#       num_feature_maps: number of filters in the convolution (decides the dimension of the output)
# Outputs:
#       x: the output from the dense block
def dense_block(x, num_layers, num_feature_maps):

    for i in range(num_layers):
        # bn = tf.keras.layers.BatchNormalization()(x)
        conv = tf.keras.layers.Conv2D(num_feature_maps, (3,3), padding='same', activation='relu')(x)

        if i == 0:
            x = conv
        else:
            x = tf.keras.layers.concatenate([conv, x])

    return x


# sr denset net cnn of 8 dense blocks
# Inputs:
#       input_shape: 
#       num_dense_block: the output from the previous block for skip connections
#       num_layers: number of convolution layers in the dense block
#       num_feature_maps: number of filters in the convolution (decides the dimension of the output)
# Outputs:
#       x: the output from the dense block
def get_Model(input_shape=(64, 64, 1), num_dense_blocks=8, num_layers=8, num_feature_maps=16):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(num_feature_maps, (3,3), padding='same', activation='relu')(inputs)

    skip_outputs = x

    for _ in range(num_dense_blocks):
        x = dense_block(skip_outputs, num_layers, num_feature_maps)
        skip_outputs = tf.keras.layers.concatenate([x, skip_outputs])
        # x = tf.keras.layers.Conv2D(x.shape[-1] // 2, (1,1), padding='same', activation='relu')(x)
        # x = tf.keras.layers.AveragePooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(256, (1,1), padding='same', activation='relu')(skip_outputs)            # bottleneck layer to reduce to 256
    x = tf.keras.layers.Conv2DTranspose(256, (3,3), strides=2, padding='same', activation='relu')(x)   # deconvolution to upsample
    x = tf.keras.layers.Conv2D(1, (3,3), padding='same')(x)                                            # reduce to single channel output

    model = tf.keras.models.Model(inputs, x)
    return model