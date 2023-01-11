import time
import numpy as np
import tensorflow as tf
import os
from img_handling import load_single_img, load_and_crop
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten


def limit_vram_usage():
    """
    Prevents tensorflow from using entire GPU VRAM
    :return: None
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def to_class(predictions) -> str:
    index = np.argmax(predictions)
    label = labels[index]

    return label


def predict_one(model, path):
    img = load_single_img(path)

    predictions = model.predict(img)

    print('Boop! Beep! Boop!')
    time.sleep(0.15)
    print(f'The machine thinks that this is {to_class(predictions)}')


def predict_batch(model, path):
    img_batch = load_and_crop(path)

    predictions = []
    for img in img_batch:
        prediction = model.predict(img)
        predictions.append(to_class(prediction))

    predictions = ', '.join(predictions)
    print('Boop! Beep! Boop!')
    time.sleep(0.25)
    print('The machine thinks the image consists of the following emojis:')
    print(predictions)


if __name__ == '__main__':

    limit_vram_usage()

    data_dir = os.path.join('data')

    # Load the data (shuffles it by default)
    train_dir = os.path.join(data_dir, 'train')
    data = keras.utils.image_dataset_from_directory(train_dir)

    # Make labels global so they dont need to be passed into a function
    global labels
    labels = data.class_names
    num_classes = len(labels)

    # Scale the color values down to improve neural network efficiency
    data = data.map(lambda img, label: (img / 255, label))

    # Split the data
    train_size = int(len(data) * .7) + 1
    val_size = int(len(data) * .2)
    test_size = int(len(data) * .1)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    model = keras.models.Sequential([
        # 16 filters, 3x3px filter size, 1px stride between filters
        Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)),
        MaxPool2D(),

        # 32 filters, 3x3px filter size, 1px stride between filters
        Conv2D(32, (3, 3), 1, activation='relu'),
        MaxPool2D(),

        # 16 filters, 3x3px filter size, 1px stride between filters
        Conv2D(16, (3, 3), 1, activation='relu'),
        MaxPool2D(),

        Flatten(),

        Dense(256, activation='relu'),

        # Output layer
        Dense(num_classes, name='outputs')
    ])

    # It is important to use loss function that is compatible with number of classes or labels that we have,
    # for example, initially I trued using BinaryCrossentropy (which doesn't make sense as there are more than
    # two labels)
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    begin_time = time.time()
    history = model.fit(train, epochs=100, validation_data=val, verbose=1)
    print(f'The model took {time.time() - begin_time} seconds to train.')

    #
    # Using the trained network
    #

    while True:
        print('Would you like to enter image of a single emoji or a collage of multiple emojis? single/collage')
        img_type = input()

        if img_type.lower() != 'collage' and img_type.lower() != 'single':
            print('Invalid input, please try again.')
            time.sleep(0.5)

        print('Please enter the path to image: ')
        img_path = input()

        if img_type.lower() == 'single':
            predict_one(model, img_path)

        else:
            predict_batch(model, img_path)