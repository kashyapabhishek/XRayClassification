from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.losses import BinaryCrossentropy
from data_ingestion.load_train_data import LoadTrainData
from data_ingestion.load_val_data import LoadValData
from data_ingestion.load_test_data import LoadTestData
from enums.image_info import ImageInfo
from preprocessing.preprocessing import Preprocessing
from tensorflow.keras.applications import mobilenet_v2, MobileNetV2
import tensorflow as tf
import os
from tensorflow.keras.layers.experimental import preprocessing


class TransferLearning(object):

    def __init__(self):
        self.model = models.Sequential()
        self.obj_train_data = LoadTrainData()
        self.obj_val_data = LoadValData()
        self.obj_test_data = LoadTestData()
        self.train_images = self.obj_train_data.data
        self.test_images = self.obj_test_data.data
        self.val_images = self.obj_val_data.data
        self.data_augmentation = None
        self.preprocessing = Preprocessing()

    def autotune_model(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.train_images = self.train_images.prefetch(buffer_size=AUTOTUNE)
        self.val_images = self.val_images.prefetch(buffer_size=AUTOTUNE)
        self.test_images = self.test_images.prefetch(buffer_size=AUTOTUNE)

    def create_model(self):
        inputs = Input(shape=ImageInfo.IMAGE_SHAPE.value)
        data_augmentation = tf.keras.Sequential([
            preprocessing.Rescaling(1. / 125)
        ])
        x = data_augmentation(inputs)
        x = mobilenet_v2.preprocess_input(x)
        x = MobileNetV2(
            input_shape=ImageInfo.IMAGE_SHAPE.value,
            include_top=False,
            weights='imagenet')(x)
        x.trainable = True
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        self.model = tf.keras.Model(inputs, outputs)

    def compile_model(self):
        base_learning_rate = 0.0001
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])

    def fit_model(self):
        self.model.fit(
            self.train_images,
            validation_data=self.val_images,
            epochs=ImageInfo.EPOCHS.value
        )

    def train(self):
        self.autotune_model()
        self.create_model()
        self.compile_model()
        self.fit_model()
        self.model.save(os.path.join(os.getcwd(), 'models/MobileNetV2/'))

    def load_model(self):
        self.model = models.load_model(os.path.join(os.getcwd(), 'models/MobileNetV2/'))
        print('model loaded')

    def evaluate(self):
        self.load_model()
        self.model.evaluate(self.test_images, verbose=2)