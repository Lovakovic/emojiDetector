import tensorflow as tf
import os
from tensorflow import keras
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten


def limit_vram_usage():
    """
    Prevents tensorflow from using entire gpu VRAM
    :return: None
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_labels() -> dict[int, str]:
    """
    Loads labels from a csv file in data dir
    :return: Labels dict containing image name (int) and its name
    """
    labels = {}

    with open(os.path.join('data', 'labels.csv'), 'r') as labels_file:
        lines = labels_file.readlines()

        for line in lines:
            tokens = line.split(',')
            labels[int(tokens[0])] = tokens[1].replace('\n', '')

    return labels


if __name__ == '__main__':
    limit_vram_usage()

    # Load the label for each image
    img_labels = load_labels()

    # Load the data (shuffles it by default)
    train_dir = os.path.join('data', 'train')
    data = keras.utils.image_dataset_from_directory(train_dir)

    # iterator = train_data.as_numpy_iterator()
    # batch = iterator.next()
    # print(batch[0].shape)

    # Scale the color values down to improve neural network efficiency
    data = data.map(lambda img, label: (img / 255, label))

    train_size = int(len(data) * .7) + 1
    val_size = int(len(data) * .2)
    test_size = int(len(data) * .1)
    # print(str(len(data)) + ' ' + str(train_size) + ' ' + str(val_size) + ' ' + str(test_size))

    # Split the data
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

        # 207 different emojis
        Dense(207, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    history = model.fit(train, epochs=20, validation_data=val, verbose=1)