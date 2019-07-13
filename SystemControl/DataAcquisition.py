"""
@title
    DataAcquisition.py
@description
"""
import argparse
import math
import random

import tensorflow as tf

from SystemControl import version, name, utilities


def calc_entropy(sentence):
    """
    Equation 3.49 (Shannon's Entropy) is implemented.
    """
    entropy = 0
    # There are 256 possible ASCII characters
    for character_i in range(256):
        Px = sentence.count(chr(character_i)) / len(sentence)
        if Px > 0:
            entropy += - Px * math.log(Px, 2)
    return entropy


def test_shannons_entropy():
    """
    We can quantify the amount of uncertainty in an entire probability distribution using the Shannon entropy.

    The entropy increases as the uncertainty of which character will be sent increases.

    :return:
    """
    # The telegrapher creates the "encoded message" with length 10000.
    # When he uses only 32 chars
    simple_message = "".join([chr(random.randint(0, 32)) for rand_int in range(10000)])
    # When he uses all 255 chars
    complex_message = "".join([chr(random.randint(0, 255)) for rand_int in range(10000)])

    simple_h = calc_entropy(simple_message)
    print('%0.4f' % simple_h)
    # Out[20]: 5.0426649536728

    complex_h = calc_entropy(complex_message)
    print('%f' % complex_h)
    # Out[21]: 7.980385887737537
    return


def test_tensorflow():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
    return


def main(args):
    """
    :param args: arguments passed in to control flow of operation.
    This is generally expected to be passed in over the command line.
    :return: None
    """
    if args.version:
        print('%s: VERSION: %s' % (name, version))
        return

    test_shannons_entropy()
    # test_tensorflow()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--version', '-v', action='store_true',
                        help='prints the current version and exits')
    print('-' * utilities.TERMINAL_COLUMNS)
    print(parser.prog)
    print('-' * utilities.TERMINAL_COLUMNS)

    cargs = parser.parse_args()
    main(cargs)
