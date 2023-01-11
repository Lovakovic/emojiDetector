import cv2
import numpy as np
from PIL import Image


def load_single_img(path: str, scale: (int, int) = (256, 256)) -> np.ndarray:
    """
    Loads, resizes, adds extra dimension and returns a single image from given path
    :param scale: Desired resolution of the image
    :param path: A path to image
    :return: Image in numpy array format
    """

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, scale)

    return np.expand_dims(img, 0)


def load_and_crop(path: str,
                  grid: (int, int) = (3, 2),
                  img_dim: (int, int) = (72, 72),
                  scale_to: (int, int) = (256, 256)) -> [np.ndarray]:
    """
    Load a collage of images and crops them into individual images, also adds one extra dimension to each image.
    The images must be uniformly distributed and have same dimensions. In case of a grid with 2 or more rows,
    the grid must be of rectangular shape.
    :param scale_to: Scales the cropped image to specified width and height
    :param path: Path to the image which will be cropped.
    :param grid: Grid of the image, how many individual images are there (width, height)
    :param img_dim: Dimensions of a single image.
    :return: A list of individual images from a collage in format of numpy arrays.
    """
    collage = Image.open(path)
    cropped_imgs = []

    for x in range(grid[0]):
        for y in range(grid[1]):
            img = collage.crop((x * img_dim[0], y * img_dim[1], (x + 1) * img_dim[0], (y + 1) * img_dim[1]))
            img = np.array(img)
            img = cv2.resize(img, scale_to)

            cropped_imgs.append(np.expand_dims(img, 0))

    return cropped_imgs
