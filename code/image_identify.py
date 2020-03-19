# -*- coding: utf-8 -*-

import sys
import tensorflow as tf

class ImageIdentify:

    model = None

    def load_model(self,filepath):
        print("a")
        self.model = tf.keras.models.load_model(filepath,compile=False)
        print("b")

    def predict(self,image):
        x = load_img(image, target_size=(299,299))        
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        prediction = self.model.predict(x)
        return jsonify(prediction)

if __name__ == '__main__':

    image = sys.argv[1]

    print("1")
    predict = ImageIdentify()
    print("2")
    predict.load_model("/data/corvidae/models/20200319-120518.hdf5")
    print("3")
    x = predict.predict(image)
    print("4")
    print(x)
