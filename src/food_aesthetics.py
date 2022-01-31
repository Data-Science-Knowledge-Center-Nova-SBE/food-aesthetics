import tensorflow as tf
from model import NimaMobileNet
import numpy as np
from PIL import Image
import cv2 as cv


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


    def aesthetic_score(self, path):
        photo = np.array(self._load_image(path))
        photo = tf.image.random_crop(tf.convert_to_tensor(photo / 255,
            dtype=tf.float16), (224, 224, 3))

        #photo = tf.convert_to_tensor(photo / 255, dtype=tf.float16), (224, 224, 3)

        logits = self.model(tf.expand_dims(photo, axis = 0))
        logits_scaled = tf.math.divide(logits, self.temperature)
        score = tf.nn.softmax(logits_scaled).numpy()[:, 1].item()
        return score


    def _load_image(self, path):
        """"
        Open and Resize Picture Mantaining Aspect Ratio.
        Shortest Side: 224 pixels
        """
        pic = Image.open(path)
        #pic = io.imread(path)
        width, height = pic.size
        s = max(224/width, 224/height)

        if width < height:
            pic_res = pic.resize((224, round(s*height)))
        else:
            pic_res = pic.resize((round(s*width), 224))

        return pic_res



    def _image2hsv(self, pic):
        """"
        input: image ID
        output: HSV matrix w/ shape H x W x C, C = 3 for H, S, and V respectively
        """
        return cv.cvtColor(np.float32(self._load_image(pic)), cv.COLOR_RGB2HSV)


    def brightness(self, path):
        """
        Cross Pixel Average of Value (2) dimension

        input: H x W x C HSV image
        output: brightness value range [0, 255]
        """
        image = self._image2hsv(path)
        return image[:,:,2].mean()

    def saturation(self, path):
        """
        Cross Pixel Average of Saturaion (1) dimension

        input: H x W x C HSV image
        output: saturation value range [0, 255]
        """
        image = self._image2hsv(path)
        return image[:, :, 1].mean()

    def contrast(self, path):
        """
        Cross Pixel Standard Deviation of Value (2) dimension

        input: H x W x C HSV image
        output: contrast value range [0, n]
        """
        image = self._image2hsv(path)
        return image[:, :, 2].std()

    def clarity(self, path, thresold = 0.7, scaler = 255):
        """
        Proportion of Normalized Value (2) Pixels that Exceed the Thresold (0.7)

        input: H x W x C HSV image
        output: clarity value range [0, 1]
        """
        image = self._image2hsv(path)
        h, w, c = image.shape
        return np.sum(image[:,:,2] / scaler > thresold) / (h * w)

    def warm(self, path):
        """"
        Proporion of Warm Hue (<30, >110) Pixels

        input: H x W x C HSV image
        output: warm value range [0, 1]
        """
        image = self._image2hsv(path)
        h, w, c = image.shape
        return np.sum((image[:, :, 0] < 30) | (image[:, :, 0] > 110)) / (h * w)

    def colourfulness(self, path):
        """
        Follow Hasler and Suesstrunk (2003) to compute colourfoulness

        input: H x W x C RGB image
        output: colourfoluness score
        """
        image = np.array(self._load_image(path))
        R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
        rg = R - G
        yb = 0.5 * (R+G) - B
        sigma = np.sqrt(np.square(np.std(rg)) + np.square(np.std(yb)))
        mu = np.sqrt(np.square(np.mean(rg)) + np.square(np.mean(yb)))
        c = sigma + 0.3 * mu # colourfoulness
        return c



if __name__ == '__main__':
    aes = FoodAesthetics()
    img = './test-images/image1.jpeg'
    print(aes.brightness(img))
    print(aes.saturation(img))
    print(aes.contrast(img))
    print(aes.clarity(img))
    print(aes.warm(img))
    print(aes.colourfulness(img))
