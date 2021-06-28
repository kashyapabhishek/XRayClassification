from tensorflow.keras.layers.experimental import preprocessing


class Preprocessing(object):

    def __init__(self):
        self.data_augmentation = None

    def augmentation(self):
        self.data_augmentation = [
            preprocessing.RandomFlip('horizontal'),
            preprocessing.RandomRotation(0.2),
            preprocessing.Rescaling(1./125)
        ]
        return self.data_augmentation