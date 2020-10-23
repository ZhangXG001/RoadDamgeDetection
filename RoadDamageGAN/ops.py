import tensorflow as tf
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers import variance_scaling_initializer as he_init


##################################################################################
# Layer
##################################################################################

def linear(x, units, use_bias=True, activation_fn='leaky', is_training=True, norm_fn='instance', scope='linear'):
    with tf.variable_scope(scope):
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=tf_contrib.layers.variance_scaling_initializer(), kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001), use_bias=use_bias)

        if norm_fn == 'instance' :
            x = instance_norm(x, 'instance')
        if norm_fn == 'batch' :
            x = batch_norm(x, is_training, 'batch')
        
        if activation_fn!='None':
            x = activation(x, activation_fn)
        return x
                
def conv(x, channels, kernel=3, stride=2, pad=0, activation_fn='leaky', is_training=True, norm_fn='instance', is_spectral_norm=False, use_bias=True, scope='conv_0') :
    with tf.variable_scope(scope):
        if scope.__contains__("discriminator") :
            weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        else :
            weight_init = tf_contrib.layers.xavier_initializer()

        x = tf.pad(x, [[0,0], [pad, pad], [pad, pad], [0,0]])
        
        if is_spectral_norm == True :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=None)
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001),
                                 strides=stride, use_bias=use_bias)

        if norm_fn == 'instance' :
            x = instance_norm(x, 'instance')
        if norm_fn == 'batch' :
            x = batch_norm(x, is_training, 'batch')
        if norm_fn == 'layer' :
            x = layer_norm(x, scope='layer_norm') 

        if activation_fn!='None':
            x = activation(x, activation_fn)
            
        return x

def deconv(x, channels, kernel=4, stride=2, activation_fn='leaky', is_training=True, norm_fn='instance', use_bias=True, scope='deconv_0'):
    with tf.variable_scope(scope):
        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                       kernel_size=kernel, kernel_initializer=weight_init, 
                                       kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001),
                                       strides=stride, padding='SAME', use_bias=use_bias)

        if norm_fn == 'instance' :
            x = instance_norm(x, 'instance')
        if norm_fn == 'batch' :
            x = batch_norm(x, is_training, 'batch')
        if norm_fn == 'layer' :
            x = layer_norm(x, scope='layer_norm') 

        if activation_fn!='None':
            x = activation(x, activation_fn)

        return x
    
def gaussian_noise_layer(mu, scope='noise'):
    with tf.variable_scope(scope):
        sigma = 0.4#tf.get_variable("sigma", [1], tf.float32, tf.random_normal_initializer(stddev=0.02))
        gaussian_random_vector = tf.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
        return mu + sigma * gaussian_random_vector

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, kernel=3, stride=1, pad=1, dropout_ratio=0.0, is_training=True, norm_fn='instance', is_spectral_norm=False, use_bias=True, scope='resblock') :
    assert norm_fn in ['instance', 'batch', 'weight', 'spectral', 'None', 'layer']
    with tf.variable_scope(scope) :
        with tf.variable_scope('res1') :
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, activation_fn='relu', norm_fn=norm_fn, is_spectral_norm=is_spectral_norm, use_bias=use_bias) 
            
        with tf.variable_scope('res2') :
            x = conv(x, channels, kernel=3, stride=1, pad=1, activation_fn='None', norm_fn=norm_fn, is_spectral_norm=is_spectral_norm, use_bias=use_bias) 

        if dropout_ratio > 0.0 :
            x = tf.layers.dropout(x, rate=dropout_ratio, training=is_training)

        return x + x_init

def adaptive_resblock(x_init, channels, mu, sigma, is_spectral_norm=False, use_bias=True, scope='adaptive_resblock') :
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, activation_fn='None', norm_fn='None', is_spectral_norm=is_spectral_norm, use_bias=use_bias) 
            x = adaptive_instance_norm(x, mu, sigma)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, activation_fn='None', norm_fn='None', is_spectral_norm=is_spectral_norm, use_bias=use_bias) 
            x = adaptive_instance_norm(x, mu, sigma)

        return x + x_init
    
def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

##################################################################################
# Activation function
##################################################################################
        
def activation(x, activation_fn='leaky') :
    assert activation_fn in ['relu', 'leaky', 'tanh', 'sigmoid', 'swish', 'selu', 'None']
    if activation_fn == 'leaky':
        x = lrelu(x)

    if activation_fn == 'relu':
        x = relu(x)

    if activation_fn == 'sigmoid':
        x = sigmoid(x)

    if activation_fn == 'tanh' :
        x = tanh(x)

    if activation_fn == 'swish' :
        x = swish(x)

    if activation_fn == 'selu' :
        x = selu(x)

    return x

def lrelu(x, alpha=0.01) :
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)

def relu(x) :
    return tf.nn.relu(x)

def sigmoid(x) :
    return tf.sigmoid(x)

def tanh(x) :
    return tf.tanh(x)

def swish(x) :
    return x * sigmoid(x)

def selu(x) :
    return tf.nn.selu(x)

##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=False, scope='batch_nom') :
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

def instance_norm(x, scope='instance_nom') :
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)

def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):
    # gamma, beta = style_mean, style_std from MLP

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Sampling
##################################################################################

def down_sample(x) :
    return tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='SAME')

def up_sample(x, scale_factor=2, metod='neighbor'):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    if metod=='bilinear':
        return tf.image.resize_bilinear(x, size=new_size)
    else:
        return tf.image.resize_nearest_neighbor(x, size=new_size)

def adaptive_avg_pooling(x):
    # global average pooling
    gap = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

    return gap

def KL_divergence(x) :
#    mu = tf.reduce_mean(x)
#    sigma_2 = tf.reduce_mean(tf.square(x-mu))
#    KL_divergence = 0.5 * (tf.square(mu) + sigma_2 - tf.log(sigma_2) - 1)
#    loss = tf.reduce_mean(KL_divergence)

    loss = tf.square(tf.reduce_mean(x))

    return loss

##################################################################################
# Loss function
##################################################################################

def regularization_loss(scope_name) :
    """
    If you want to use "Regularization"
    g_loss += regularization_loss('generator')
    d_loss += regularization_loss('discriminator')
    """
    collection_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = []
    for item in collection_regularization :
        if scope_name in item.name :
            loss.append(item)

    return tf.reduce_sum(loss)

def L1_loss(x, y) :
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'hinge' :
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss

def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func.__contains__('wgan') :
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge' :
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss

def classification_loss1(logit, label) :
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
    return loss

def classification_loss2(logit, label) :
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    return loss

# def classification_loss2(label_o, label_p) :
#     loss = tf.reduce_mean(-label_o*tf.log(tf.clip_by_value(label_p,1e-10,1.0))-(1-label_o)*tf.log(tf.clip_by_value((1-label_p),1e-10,1.0)))
#     return loss

