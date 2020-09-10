from flask import Flask,request
from flask_restful import Api, Resource, reqparse, fields
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
app = Flask(__name__)
api = Api(app)

class Predict(Resource):
    def get(self):
        return "app running"
    def post(self):
        model = tf.keras.models.load_model('./model.h5')
        file = request.files['file']
        filename = secure_filename(file.filename)
        os.mkdir("tmp")
        filename="tmp/"+filename
        file.save(filename)#savefile
        img = tf.keras.preprocessing.image.load_img(filename,target_size=(224,224,3))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x/255.0
        pred=model.predict(x)
        os.remove(filename)#deletefile
        os.rmdir("tmp")
        print(pred[0][0])
        ret_obj={'prediction':str(pred[0][0])}
        return ret_obj

api.add_resource(Predict, "/")
if __name__ == "__main__":
    app.run(debug=True)