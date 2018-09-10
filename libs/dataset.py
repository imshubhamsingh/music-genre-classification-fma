from random import shuffle
import os
from PIL import Image
import numpy as np

root_path = os.path.join(os.path.dirname(__file__), os.path.pardir)


def load_spectrogram(path, image_size):
    spectrogram = Image.open(path)
    spectrogram = spectrogram.resize(
        (image_size, image_size), resample=Image.ANTIALIAS)
    spectrogram_data = np.asarray(
        spectrogram, dtype=np.uint8).reshape(image_size, image_size, 1)
    return spectrogram_data / 255


def create_dataset_from_slices(spectrograms_per_genre, genres, slice_size, validation_ratio, test_ratio):
    data = []
    for genre in genres:
        print("-> Adding {}...".format(genre))
        filenames = os.listdir(os.path.join(
            root_path, "preprocessing", "slices", genre))
        filenames = [
            filename for filename in filenames if filename.endswith('.png')]
        filenames = filenames[:spectrograms_per_genre]
        shuffle(filenames)

        # Add data (X,y)
        for filename in filenames:
            img_data = load_spectrogram(os.path.join(
                root_path, "preprocessing", "slices", genre, filename), slice_size)
            label = [1. if genre == g else 0. for g in genres]
            data.append((img_data, label))

    # Shuffle data
    shuffle(data)

    # Extract X and y
    x, y = zip(*data)

    # Split data
    validation_len = int(len(x) * validation_ratio)
    test_len = int(len(x) * test_ratio)
    train_len = len(x) - (validation_len + test_len)

    # Prepare for Tflearn at the same time
    train_x = np.array(x[:train_len]).reshape([-1, slice_size, slice_size, 1])
    train_y = np.array(y[:train_len])
    validation_x = np.array(
        x[train_len:train_len + validation_len]).reshape([-1, slice_size, slice_size, 1])
    validation_y = np.array(y[train_len:train_len + validation_len])
    test_x = np.array(x[-test_len:]).reshape([-1, slice_size, slice_size, 1])
    test_y = np.array(y[-test_len:])
    return train_x, train_y, validation_x, validation_y, test_x, \
        test_y


def create_test_dataset_from_slices(path, slice_size):
    data = []
    filenames = os.listdir(os.path.join(root_path, path))
    filenames = [
        filename for filename in filenames if filename.endswith('.png')]
    for filename in filenames:
        img_data = load_spectrogram(os.path.join(
            root_path, path, filename), slice_size)
        data.append(img_data)

    return np.array(data).reshape([-1, slice_size, slice_size, 1])
