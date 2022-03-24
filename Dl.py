# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:37:13 2019

@author: user
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import regularizers
from keras.layers.core import Dropout

classifier = Sequential()

classifier.add(Convolution2D(32 , 3 , 3 , input_shape=(64, 64 , 3) , activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2 , 2)))

classifier.add(Convolution2D(32 , 3 , 3 , activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2 , 2)))

classifier.add(Convolution2D(32 , 3 , 3 , activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2 , 2)))

classifier.add(Convolution2D(32 , 3 , 3 , activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2 , 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128 , activation='relu'))

classifier.add(Dense(output_dim = 1 , activation='sigmoid'))

classifier.add(Dropout(0.25))

classifier.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

testing_set = test_datagen.flow_from_directory(
        'test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

results = classifier.fit_generator(
        training_set,
        steps_per_epoch=2637,
        epochs=30,
        validation_data=testing_set,
        validation_steps=660)

classifier.summary()

pip install Pillow

 pip install scipy==1.1.0
 
 import scipy.misc
imgs = scipy.misc.imread('test/benign/1.jpg')

imgs = scipy.misc.imresize(imgs,(64 , 64))

classifier.predict_classes(imgs.reshape(1,64,64,3))

true_classes = testing_set.classes
class_labels = list(testing_set.class_indices.keys())   

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#Graphing our training and validation
acc = results.history['acc']
val_acc = results.history['val_acc']
loss = results.history['loss']
val_loss = results.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()

#Confution Matrix and Classification Report (To check an accuracy)
from sklearn.metrics import confusion_matrix
batch_size = 32
num_of_test_samples = 660
Y_pred = classifier.predict_generator(testing_set, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(testing_set.classes, y_pred)
print('Confusion Matrix')
print(cm)

print('Classification Report')
target_names = ['Benign', 'Maliginent']
print(classification_report(testing_set.classes, y_pred, target_names=target_names))

import seaborn as sns

#print the confusion metrix based on the test values
sns.set()
get_ipython().run_line_magic('matplotlib','inline')
sns.heatmap(cm.T,square=True,annot=True,fmt='d',cbar=False)
plt.xlabel('True value')
plt.ylabel('Predicted value')

import tensorflow as tf

import os
import h5py
f = h5py.File('model.h5','w')

classifier.save(f)

converter = tf.lite.TFLiteConverter.from_keras_model_file( 'model.h5' ) 
model = converter.convert()

open("model.tflite" , "wb").write(model)

import os
import h5py
f = h5py.File('skin_cancer_cnn.hdf5','w')
#classifier.load_weights('best_weights.hdf5')
classifier.save(f)

import tensorflow as tf
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph
    
from keras import backend as K
pb_filename = 'model.pb'
wkdir = '/content'
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in classifier.outputs])
tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)

import tensorflow as tf
from tensorflow import keras

keras_model = 'model.h5'
classifier.save(keras_model)

new_model = keras.models.load_model(keras_model)
new_model.summary()


