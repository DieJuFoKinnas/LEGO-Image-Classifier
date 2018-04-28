import numpy as np
import tensorflow as tf
from config import cfg
from capslayer import layers

tf.logging.set_verbosity(tf.logging.INFO)


def get_margin_loss(labels, predictions):
    with tf.name_scope('margin_loss'):
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - predictions))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., predictions - cfg.m_minus))

        # reshape: [batch_size, num_label, 1, 1] => [batch_size, num_label]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # calc T_c: [batch_size, num_label]
        # T_c = Y, is my understanding correct? Try it.
        T_c = labels
        # [batch_size, num_label], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        return tf.reduce_mean(tf.reduce_sum(L_c, axis=1))


def get_reconstruction_loss(input_layer, reconstruction):
    with tf.name_scope('reconstruction_loss'):
        original = tf.reshape(input_layer, shape=(cfg.batch_size, -1))
        squared = tf.square(reconstruction - original)
        return tf.reduce_mean(squared)


def caps_model_fn(features, labels, mode):
    """Model function for CapsNet."""
    height = 28
    width = 28
    classes = 10

    input_layer = tf.reshape(features["x"], [-1, height, width, 1])
    labels = tf.one_hot(labels, depth=classes, axis=1, dtype=tf.float32)

    # returns shape [batch_size, 20, 20, 256]
    conv1 = tf.layers.conv2d(input_layer, 256, 9, strides=1, padding='VALID')
    # returns primaryCaps: [batch_size, 1152, 8, 1], activation: [batch_size, 1152]
    with tf.variable_scope('primary_caps'):
        primary_caps, activation = layers.primaryCaps(conv1, filters=32, kernel_size=9, strides=2, out_caps_shape=[8, 1])

    # return digitCaps: [batch_size, num_label, 16, 1], activation: [batch_size, num_label]
    with tf.variable_scope('digit_caps'):
        # primary_caps = tf.reshape(primary_caps, shape=[cfg.batch_size, -1, 8, 1])
        digit_caps, activation = layers.fully_connected(primary_caps, activation, num_outputs=10, out_caps_shape=[16, 1],
                                                        routing_method='DynamicRouting')

    # Reconstruct the MNIST images with 3 FC layers
    # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
    with tf.variable_scope('decoder'):
        masked_caps = tf.multiply(digit_caps, tf.reshape(labels, (-1, 10, 1, 1)))
        active_caps = tf.reshape(masked_caps, shape=(cfg.batch_size, -1))
        fc1 = tf.contrib.layers.fully_connected(active_caps, num_outputs=512)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
        reconstruction = tf.contrib.layers.fully_connected(fc2, num_outputs=height * width, activation_fn=tf.sigmoid)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=activation, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
        "probabilities": tf.nn.softmax(activation, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # The paper uses sum of squared error as reconstruction error, but we
    # have used reduce_mean in `# 2 The reconstruction loss` to calculate
    # mean squared error. In order to keep in line with the paper,the
    # regularization scale should be 0.0005*784=0.392
    margin_loss = get_margin_loss(labels, activation)
    reconstruction_loss = get_reconstruction_loss(input_layer, reconstruction)
    loss = margin_loss + cfg.regularization_scale * reconstruction_loss

    tf.summary.scalar('train/margin_loss', margin_loss)
    tf.summary.scalar('train/reconstruction_loss', reconstruction_loss)
    tf.summary.scalar('train/total_loss', loss)
    recon_img = tf.reshape(reconstruction, shape=(cfg.batch_size, height, width, 1))
    tf.summary.image('reconstruction_img', recon_img)
    tf.summary.histogram('activation', activation)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(_):
    # TODO: I think there are some things missing to copy from the CapsLayer repo
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    config = tf.estimator.RunConfig(save_checkpoints_steps=5000)
    mnist_classifier = tf.estimator.Estimator(model_fn=caps_model_fn, model_dir="mnist_capsnet_model", config=config)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)

    # Train the model
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #   x={"x": train_data},
    #   y=train_labels,
    #   batch_size=32,
    #   num_epochs=None,
    #   shuffle=True)
    # mnist_classifier.train(
    #   input_fn=train_input_fn,
    #   steps=300,
    #   hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      batch_size=32,
      num_epochs=1,
      shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
