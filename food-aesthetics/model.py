import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.mobilenet import MobileNet


class NimaMobileNet(tf.keras.Model):
    def __init__(self, training = True):

        super(NimaMobileNet, self).__init__()
        self.training = training
        self.base_model = MobileNet((None, None, 3), alpha=1, include_top=False,
            pooling='avg', weights=None)
        if self.training:
            self.x = Dropout(0.25)(self.base_model.output)
        else:
            self.x = self.base_model.output
        self.x = Dense(10, activation='relu')(self.x)
        self.model = Model(self.base_model.input, self.x)

        # Add New Layers
        self.fc_last = Dense(2)

    def call(self, x):
        x = self.model(x)
        return self.fc_last(x)

if __name__ == '__main__':
    model = NimaMobileNet(training = False)
    model.build((1, 224, 224, 3))
    model.load_weights('./weights/trained_weights.h5')
