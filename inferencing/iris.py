
import numpy as np
import tensorflow as tf
import os

path="./models/model.h5"
if not os.path.isfile(path):
    path="./inferencing/models/model.h5"
model = tf.keras.models.load_model(path)

# [0.07471338 0.07941484 0.0885422  0.08627246]
def predict_iris(raw_input):
    prediction_raw = model.predict(np.array(raw_input))
    # prediction value is formated in [class, score]
    prediction=[[np.argmax(pr),max(pr)] for pr in prediction_raw]

    return prediction

