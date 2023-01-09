import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
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


def to_class(prediction) -> str:
    # Unwrap it
    prediction = prediction[0]
    most_certain = prediction[0]
    index = 0

    for i in range(len(prediction)):
        certainty = prediction[i]

        if certainty > most_certain:
            most_certain = certainty
            index = i

    return labels[index]


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

    history = model.fit(train, epochs=20, validation_data=val, verbose=1)

    # img = cv2.imread(os.path.join(data_dir, 'test', 'grinning face', 'twitter.png'))
    # resize = tf.image.resize(img, (256, 256))
    # Since model expects a batch of images we have to
    # yhat = model.predict(np.expand_dims(resize/255, 0))
    # print(to_class(yhat))

    img = keras.utils.load_img(
        os.path.join(data_dir, 'test', 'alien', 'facebook.jpeg'),
        target_size=(256, 256)
    )

    np_array_img = tf.keras.preprocessing.image.img_to_array(img)
    plt.imshow(np_array_img)
    plt.show()

    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(f'Boop! Beep! Boop! The machine thinks that this is {labels[np.argmax(score)]} with {100 * np.max(score)}%'
          f' confidence.')