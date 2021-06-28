from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.losses import BinaryCrossentropy
from data_ingestion.load_train_data import LoadTrainData
from data_ingestion.load_val_data import LoadValData
from enums.image_info import ImageInfo


class CustomCnn(object):

    def __init__(self):
        self.model = models.Sequential()
        self.obj_train_data = LoadTrainData()
        self.obj_val_data = LoadValData()
        self.train_images = self.obj_train_data.data
        self.val_images = self.obj_val_data.data

    def create_model(self):
        self.model.add(Input(shape=ImageInfo.IMAGE_SHAPE.value))
        self.model.add(Conv2D(64, activation='relu', kernel_size=(5, 5), strides=(2, 2)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        self.model.add(Conv2D(128, activation='relu', kernel_size=(5, 5), strides=(2, 2)))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

    def compile_model(self):
        self.model.compile(
            loss=BinaryCrossentropy(from_logits=False),
            optimizer='adam',
            metrics=['accuracy']
        )

    def fit_model(self):
        self.model.fit(
            self.train_images,
            validation_data=self.val_images,
            batch_size=ImageInfo.BATCH_SIZE.value,
            epochs=ImageInfo.EPOCHS.value
        )

    def train(self):
        self.create_model()
        self.compile_model()
        self.fit_model()
        self.model.save()