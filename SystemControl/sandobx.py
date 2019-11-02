"""
@title
@description
"""
import tensorflow as tf


def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    first_img_train = x_train[0]
    first_img_test = x_test[0]

    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    return


if __name__ == '__main__':
    main()
