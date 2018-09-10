import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


def create_model(genre_classes, image_size):
    print("Creating model...")
    c = input_data(shape=[None, image_size, image_size, 1], name='input')

    c = conv_2d(c, 64, 2, activation='elu', weights_init="Xavier")
    c = max_pool_2d(c, 2)

    c = conv_2d(c, 128, 2, activation='elu', weights_init="Xavier")
    c = max_pool_2d(c, 2)

    c = conv_2d(c, 256, 2, activation='elu', weights_init="Xavier")
    c = max_pool_2d(c, 2)

    c = conv_2d(c, 512, 2, activation='elu', weights_init="Xavier")
    c = max_pool_2d(c, 2)

    c = fully_connected(c, 1024, activation='elu')
    c = dropout(c, 0.5)

    c = fully_connected(c, genre_classes, activation='softmax')
    c = regression(c, optimizer='rmsprop',
                   loss='categorical_crossentropy')

    model = tflearn.DNN(c, tensorboard_verbose=3,
                        tensorboard_dir='./tflearnLogs/')
    return model
