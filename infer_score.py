import tensorflow as tf
from src.model import NimaMobileNet
import numpy as np
from PIL import Image


class FoodAesthetics:
    def __init__(self):

        super(FoodAesthetics, self).__init__()

        # workaround: set a random seed
        tf.random.set_seed(46)

        self.__batch_size = 1
        self.temperature = 1.536936640739441
        self.model = NimaMobileNet('./weights/pre_trained_weights.h5',
            training=False)
        self.model.build((self.__batch_size, 224, 224, 3))
        self.model.load_weights('./weights/trained_weights.h5')


    def __load_image(self, path):
        """"
        Open and Resize Picture Mantaining Aspect Ratio.
        Shortest Side: 224 pixels
        """
        pic = Image.open(path)
        width, height = pic.size
        s = max(224/width, 224/height)

        if width < height:
            pic_res = pic.resize((224, round(s*height)))
        else:
            pic_res = pic.resize((round(s*width), 224))

        return pic_res


    def aesthetic_score(self, path):
        photo = np.array(self.__load_image(path))
        photo = tf.image.random_crop(tf.convert_to_tensor(photo / 255,
            dtype=tf.float16), (224, 224, 3))

        #photo = tf.convert_to_tensor(photo / 255, dtype=tf.float16), (224, 224, 3)

        logits = self.model(tf.expand_dims(photo, axis = 0))
        logits_scaled = tf.math.divide(logits, self.temperature)
        score = tf.nn.softmax(logits_scaled).numpy()[:, 1].item()
        return score


if __name__ == '__main__':
    aes = FoodAesthetics()
    print(aes.aesthetic_score('./test-images/image1.jpeg'))
