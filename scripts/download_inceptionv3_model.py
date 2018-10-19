from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

inception_model = InceptionV3(weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
inception_model.save('inception.h5')
