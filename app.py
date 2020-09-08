from flask import Flask,request
from flask_restful import Api, Resource, reqparse, fields
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
app = Flask(__name__)
api = Api(app)

class Predict(Resource):
    def get(self):
        return "app running"
    def post(self):
        file=request.files['file']
        filename = secure_filename(file.filename)
        img = tf.keras.preprocessing.image.load_img(filename,target_size=(224,224,3))
        model = tf.keras.models.load_model('./model.h5')
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x/255.0
        pred=model.predict(x)
        print(pred[0][0])
        ret_obj={'prediction':str(pred[0][0])}
        return ret_obj

api.add_resource(Predict, "/")
if __name__ == "__main__":
    app.run(debug=True)