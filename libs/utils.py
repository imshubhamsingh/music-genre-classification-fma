import ast
import pandas as pd
from pandas.api.types import CategoricalDtype
import librosa as librosa
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import shutil
from tqdm import tqdm
from subprocess import Popen, PIPE, STDOUT

root_path = os.path.join(os.path.dirname(__file__), os.path.pardir)


def load(filepath):
    tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

    COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
               ('track', 'genres'), ('track', 'genres_all')]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
               ('album', 'date_created'), ('album', 'date_released'),
               ('artist', 'date_created'), ('artist', 'active_year_begin'),
               ('artist', 'active_year_end')]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ('small', 'medium', 'large')
    tracks['set', 'subset'] = tracks['set', 'subset'].astype(
        CategoricalDtype(categories=SUBSETS, ordered=True))

    COLUMNS = [('track', 'genre_top'), ('track', 'license'),
               ('album', 'type'), ('album', 'information'),
               ('artist', 'bio')]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype('category')

    return tracks


def load_music(audio, sampling_rate):
    try:
        wave, fs = librosa.load(audio, sr=sampling_rate)
        return wave
    except:
        track_id = os.path.splitext(os.path.basename(audio))[0].lstrip('0')
        print("\nNo music in track_id", track_id)


def get_all_music(path, labels, X, Y):
    print("Getting music files and converting it into spectrogram ...")
    music_files = librosa.util.find_files(directory=path, recurse=True)
    t = tqdm(music_files, desc='Bar desc', leave=True)
    if not os.path.exists(os.path.join(os.path.dirname(__file__), os.path.pardir, "tmp")):
        os.makedirs(os.path.join(os.path.dirname(
            __file__), os.path.pardir, "tmp"))
    for audio in t:
        t.set_description("file: {}".format(os.path.basename(audio)))
        track_id = os.path.splitext(os.path.basename(audio))[0].lstrip('0')
        genre = labels.loc[labels['track_id']
                           == int(track_id)]['genre'].values[0]
        spectrogram(audio, int(track_id), genre, X, Y)
        t.refresh()
    print("Spectrogram created")
    shutil.rmtree(os.path.join(root_path, "tmp"))


def get_music(path, sampling_rate):
    m = load_music(path, sampling_rate)
    track_id = os.path.splitext(os.path.basename(path))[0].lstrip('0')
    if m is not None:
        return {track_id: m}
    else:
        return False


def get_music_genre(path, subset):
    tracks = load(path)
    subset = tracks.index[tracks['set', 'subset'] <= subset]
    labels = tracks.loc[subset, ('track', 'genre_top')]
    labels.name = 'genre'
    labels.to_csv('preprocessing/train_labels.csv', header=True)


def get_genre():
    tracks = pd.read_csv(os.path.join(
        root_path, "preprocessing", "train_labels.csv"))
    return tracks['genre'].unique()


def spectrogram(music_path, track_id, genre, X, Y):
    remove_stereo = "sox '{}' 'tmp/{:06d}.mp3' remix 1-2".format(
        music_path, track_id)
    process = Popen(remove_stereo, shell=True, stdin=PIPE,
                    stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=root_path)
    output, errors = process.communicate()
    if errors:
        print(errors)
    # Create spectrogram
    if not os.path.exists(os.path.join(os.path.dirname(__file__), os.path.pardir, "spectrograms", genre)):
        os.makedirs(os.path.join(os.path.dirname(__file__),
                                 os.path.pardir, "spectrograms", genre))
    generate_spectrogram = "sox 'tmp/{:06d}.mp3' -n spectrogram -Y {} -X {} -m -r -o \
                            'spectrograms/{}/{:06d}.png'".format(
        track_id,
        Y,
        X,
        genre,
        track_id,
    )

    process = Popen(generate_spectrogram, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                    close_fds=True, cwd=root_path)
    output, errors = process.communicate()
    if errors:
        print(errors)


def slice_spectrograms(path, desired_size):
    spectrograms_path = os.path.join(root_path, path)
    print("Slicing All Spectrograms")
    t = tqdm(os.listdir(spectrograms_path), desc='Bar desc', leave=True)
    for genre_folder in t:
        for spectrogram in os.listdir(os.path.join(spectrograms_path, genre_folder)):
            if spectrogram.endswith('png'):
                t.set_description(
                    "file: {}/{}".format(genre_folder, spectrogram))
                slice_(os.path.join(root_path, path,
                                    genre_folder, spectrogram), desired_size)
                t.refresh()
    print("Spectrogram slice created")


def test_spectrogram(music_path):
    track_id = int(music_path.split('/')[-1].partition('.')[0])
    if not os.path.exists(os.path.join(os.path.dirname(__file__), os.path.pardir, "test", "tmp")):
        os.makedirs(os.path.join(os.path.dirname(__file__),
                                 os.path.pardir, "test", "tmp"))
    remove_stereo = "sox '{}' 'test/tmp/{:06d}.mp3' remix 1-2".format(
        music_path, track_id)
    process = Popen(remove_stereo, shell=True, stdin=PIPE,
                    stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=root_path)
    output, errors = process.communicate()
    if errors:
        print(errors)
    # Create spectrogram
    if not os.path.exists(os.path.join(os.path.dirname(__file__), os.path.pardir, "test", "spectrograms", "{:06d}".format(track_id))):
        os.makedirs(os.path.join(os.path.dirname(__file__), os.path.pardir,
                                 "test", "spectrograms", "{:06d}".format(track_id)))
    generate_spectrogram = "sox 'test/tmp/{:06d}.mp3' -n spectrogram -Y 200 -X 20 -m -r -o \
                                'test/spectrograms/{:06d}/{:06d}.png'".format(
        track_id,
        track_id,
        track_id,
    )

    process = Popen(generate_spectrogram, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                    close_fds=True, cwd=root_path)
    output, errors = process.communicate()
    if errors:
        print(errors)
    shutil.rmtree(os.path.join(root_path, "test", "tmp"))


def test_slice_(img_path, desired_size):
    img = Image.open(os.path.join(root_path, img_path))
    width, height = img.size
    samples_size = int(width / desired_size)
    print(samples_size, width)
    track_id = int(img_path.split('/')[-1].strip('.png'))
    slice_path = 'test/slice/{:06d}'.format(track_id)

    if not os.path.exists(os.path.join(root_path, slice_path)):
        os.makedirs(os.path.join(root_path, slice_path))

    for i in range(samples_size):
        start_pixel = i * desired_size
        img_tmp = img.crop((start_pixel, 1, start_pixel +
                            desired_size, desired_size + 1))
        save_path = os.path.join(root_path, slice_path)
        img_tmp.save(save_path+"/{}_{}.png".format(track_id, i))


def slice_(img_path, desired_size):
    img = Image.open(img_path)
    width, height = img.size
    samples_size = int(width / desired_size)
    genre = img_path.split('/')[-2]
    track_id = img_path.split('/')[-1]
    slice_path = 'preprocessing/slices/{}'.format(genre)

    if not os.path.exists(os.path.join(root_path, slice_path)):
        os.makedirs(os.path.join(root_path, slice_path))

    for i in range(samples_size):
        start_pixel = i * desired_size
        img_tmp = img.crop((start_pixel, 1, start_pixel +
                            desired_size, desired_size + 1))
        save_path = os.path.join(root_path, slice_path)
        img_tmp.save(save_path+"/{}_{}.png".format(track_id.rstrip('.png'), i))


def confusion_matrix(labels, predictions, classes):
    confusion_mat = tf.confusion_matrix(labels, predictions, num_classes=classes, dtype=tf.int32, name=None,
                                        weights=None)
    print(np.matrix(confusion_mat))


if __name__ == "__main__":
    test_slice_('test/spectrograms/078038/078038.png', 198)
