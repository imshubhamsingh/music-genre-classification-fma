import os
import uuid

root_path = os.path.join(os.path.dirname(__file__), os.path.pardir)


def train(model, train_x, train_y, validation_x, validation_y, batch_size, epoch):
    print("Training the model")
    model.fit(train_x, train_y, n_epoch=epoch, batch_size=batch_size, shuffle=True,
              validation_set=(validation_x, validation_y), snapshot_step=100)
    print("Model trained")
    print("Saving the weights...")
    id = uuid.uuid1()
    model.save(os.path.join(root_path, 'models',
                            'model-{}'.format(id.hex)))
    print("Weights saved")
    return id.hex


def test(model, test_x, test_y, id):
    print("Loading weights from model {}...".format(id))
    model.load(os.path.join(root_path, 'models', 'model-{}'.format(id)))
    print("Weights loaded")

    test_accuracy = model.evaluate(test_x, test_y)[0]
    print("Test accuracy: {} ".format(test_accuracy))
