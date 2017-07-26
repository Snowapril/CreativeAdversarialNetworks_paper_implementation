import tensorflow as tf


def leaky_relu(x, alpha=0.2, max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32), tf.cast(max_value, dtype=tf.float32))

    x -= tf.constant(alpha, dtype=tf.float32) * negative_part

    return x



def generator(z, reuse=False, name="gen"):
    """DCGAN
     z ∈ R -> 100 → 4 × 4 × 1024 → 8 × 8 × 1024 → 16 × 16 × 512 →
    32 × 32 × 256 → 64 × 64 × 128 → 128 × 128 × 64 → 256 × 256 × 3 """
    with tf.variable_scope(name, reuse=reuse) as scope:
        with tf.variable_scope("reshape"):
            inputs = tf.layers.dense(z, 1024 * 4 * 4, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            inputs = tf.reshape(inputs, [-1, 4, 4, 1024])
            inputs = tf.layers.batch_normalization(inputs)
            inputs = tf.nn.relu(inputs)

        with tf.variable_scope("convolution"):
            # a series of four fractionall-stride convolutions which is wrongly called "deconvolutions"
            layer1 = tf.layers.conv2d_transpose(inputs, 1024, kernel_size=[5,5] ,strides=(2,2),padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer1 = tf.layers.batch_normalization(layer1)
            layer1 = tf.nn.relu(layer1)

            layer2 = tf.layers.conv2d_transpose(layer1, 512, kernel_size=[5,5] ,strides=(2,2),padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer2 = tf.layers.batch_normalization(layer2)
            layer2 = tf.nn.relu(layer2)

            layer3 = tf.layers.conv2d_transpose(layer2, 256, kernel_size=[5,5] ,strides=(2,2),padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer3 = tf.layers.batch_normalization(layer3)
            layer3 = tf.nn.relu(layer3)

            layer4 = tf.layers.conv2d_transpose(layer3, 128, kernel_size=[5,5] ,strides=(2,2),padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer4 = tf.layers.batch_normalization(layer4)
            layer4 = tf.nn.relu(layer4)

            layer5 = tf.layers.conv2d_transpose(layer4, 64, kernel_size=[5,5] ,strides=(2,2),padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer5 = tf.layers.batch_normalization(layer5)
            layer5 = tf.nn.relu(layer5)

            layer6 = tf.layers.conv2d_transpose(layer5, 3, kernel_size=[5,5] ,strides=(2,2),padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

        with tf.variable_scope("result"):
            result = tf.nn.tanh(layer6)

        return result


def Discriminator(x, K,reuse=False, name="disc"):
    """
     256*256*3 → 128*128*32 → 64*64*64 → 32*32*128 → 16*16*256 → 8*8*512 → 4*4*512
    """
    with tf.variable_scope(name, reuse=reuse) as scope:
        with tf.variable_scope("convolution"):
            layer1 = tf.layers.conv2d(x, 32, kernel_size=[4,4], strides=(2,2), padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer1 = leaky_relu(layer1)

            layer2 = tf.layers.conv2d(layer1, 64, kernel_size=[4,4], strides=(2,2), padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer2 = tf.layers.batch_normalization(layer2)
            layer2 = leaky_relu(layer2)

            layer3 = tf.layers.conv2d(layer2, 128, kernel_size=[4,4], strides=(2,2), padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer3 = tf.layers.batch_normalization(layer3)
            layer3 = leaky_relu(layer3)

            layer4 = tf.layers.conv2d(layer3, 256, kernel_size=[4,4], strides=(2,2), padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer4 = tf.layers.batch_normalization(layer4)
            layer4 = leaky_relu(layer4)

            layer5 = tf.layers.conv2d(layer4, 512, kernel_size=[4,4], strides=(2,2), padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer5 = tf.layers.batch_normalization(layer5)
            layer5 = leaky_relu(layer5)

            layer6 = tf.layers.conv2d(layer5, 512, kernel_size=[4,4], strides=(2,2), padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer6 = tf.layers.batch_normalization(layer6)

            body_result = leaky_relu(layer6)

        R_result = discriminator_R(body_result, reuse=reuse)
        C_result = discriminator_C(body_result, K, reuse=reuse)

    return R_result, C_result

def discriminator_R(x, reuse=False, name="disc_R"):
    """
    4*4*512 -> 2
    """
    with tf.variable_scope(name, reuse=reuse) as scope:
        with tf.variable_scope("reshape"):
            inputs = tf.reshape(x, [-1, 4*4*512])
            dense = tf.layers.dense(inputs, 2, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            result = tf.nn.sigmoid(dense)

    return result

def discriminator_C(x, K, reuse=False, name="disc_C"):
    """
    4*4*512 → 1024 → 512 → K
    """
    with tf.variable_scope(name, reuse=reuse) as scope:
        with tf.variable_scope("reshape"):
            inputs = tf.reshape(x, [-1, 4*4*512])

            layer1 = tf.layers.dense(inputs, 1024, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer1 = leaky_relu(layer1)

            layer2 = tf.layers.dense(layer1, 512, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            layer2 = leaky_relu(layer2)

            layer3 = tf.layers.dense(layer2, K, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            result = tf.nn.sigmoid(layer3)

    return result
