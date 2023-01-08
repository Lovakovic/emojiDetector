import tensorflow as tf
import os
from tensorflow import keras


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

    train_dir = os.path.join('data', 'train')
    data = keras.utils.image_dataset_from_directory(train_dir)