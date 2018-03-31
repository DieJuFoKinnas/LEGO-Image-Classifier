import numpy as np
import tensorflow as tf
import math

tf.logging.set_verbosity(tf.logging.INFO)


def encoder(image, mode):
    # output tensor shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(inputs=image, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # output tensor shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # output tensor shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    # output tensor shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # output tensor shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    return tf.layers.dense(inputs=dropout, units=16)


def get_predictions(embedding, nce_weights, nce_biases):
    logits = tf.matmul(embedding, tf.transpose(nce_weights)) + nce_biases
    classes = tf.argmax(input=logits, axis=1)

    return {
        "classes": classes,
        "probability": tf.gather(logits, classes),
    }


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # labels is just a tensor of size [batch_size]
    # features["x"] is actually just a flattened tensor that has to be reshaped
    embedding = encoder(tf.reshape(features["x"], [-1, 28, 28, 1]), mode)

    num_sampled = 4
    vocabulary_size = 10
    embedding_size = 16

    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    if mode == tf.estimator.ModeKeys.TRAIN:
        labels = tf.expand_dims(labels, 1)
        sampled_values = tf.nn.uniform_candidate_sampler(labels, 1, num_sampled, unique=True, range_max=vocabulary_size)
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=tf.expand_dims(labels, 1),
                inputs=embedding,
                num_sampled=num_sampled,
                num_classes=vocabulary_size,
                sampled_values=sampled_values))

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    logits = tf.matmul(embedding, tf.transpose(nce_weights)) + nce_biases
    classes = tf.argmax(input=logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": classes,
            "probability": tf.gather(logits, classes, name='certainty'),
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    elif mode == tf.estimator.ModeKeys.EVAL:
        one_hot = tf.one_hot(labels, vocabulary_size, axis=1)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot, logits=logits)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=classes)
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(_):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int64)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int64)

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="mnist_nce_model")

    # tensors_to_log = {"probability": "certainty"}
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, y=train_labels, batch_size=32, num_epochs=None, shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn, steps=2000)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
