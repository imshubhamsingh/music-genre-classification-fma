import numpy as np
from libs import utils, dataset, process, model

genres = np.array(['Hip-Hop', 'Pop', 'Folk', 'Experimental', 'Rock', 'International',
                   'Electronic', 'Instrumental'])

train_x, train_y, validation_x, \
    validation_y, test_x, test_y = dataset.create_dataset_from_slices(
        2700, genres, 198, 0.15, 0.1)

classes = genres.shape[0]

DNN_model = model.create_model(classes, 198)

_id = process.train(DNN_model, train_x, train_y,
                    validation_x, validation_y, 198, 20)

process.test(DNN_model, test_x, test_y, _id)
