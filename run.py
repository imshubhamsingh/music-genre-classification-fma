import numpy as np
import pandas as pd
from libs import utils, dataset, process, model

metadata_genre_file_path = './dataset/fma_metadata/tracks.csv'
subset = 'small'
utils.get_music_genre(metadata_genre_file_path, subset)
labels = pd.read_csv('preprocessing/train_labels.csv')


genres = np.array(['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International',
                   'Electronic', 'Instrumental'])


dataset_path = 'dataset/fma_small'
utils.get_all_music(dataset_path, labels, 20, 200)
utils.slice_spectrograms("spectrograms", 198)

train_x, train_y, validation_x, \
    validation_y, test_x, test_y = dataset.create_dataset_from_slices(
        2700, genres, 198, 0.15, 0.1)

classes = genres.shape[0]

DNN_model = model.create_model(classes, 198)

_id = process.train(DNN_model, train_x, train_y,
                    validation_x, validation_y, 198, 20)

process.test(DNN_model, test_x, test_y, _id)
