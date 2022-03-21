# import the necessary packages
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neural_network import MLPClassifier
import argparse
import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
# Helper functions
# Create model
def create_model():
	model = tf.keras.models.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(128,)),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(u_names)
  ])
	model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()],)

	return model
# Freeze and save tf graph
def freeze_graph(model_dir, output_node_name):
  if not tf.gfile.Exists(model_dir):
    raise AssertionError(
      "Export failed!! Directory doesnt exist"
    )
  
  if not output_node_name:
    return -1
  
  checkpoint = tf.train.get_checkpoint_state(model_dir)
  input_checkpoint = checkpoint.model_checkpoint_path

  absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
  output_graph = absolute_model_dir+"/frozen_model.pb"

  clear_devices = True

  with tf.Session(graph = tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(input_checkpoint+'.meta',clear_devices=clear_devices)
    saver.restore(sess, input_checkpoint)

    output_graph_def
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())

print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())
u_names = len(set(data['names']))

print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# output/embeddings.pickle 
labels = np.array(labels,dtype='uint8')


print("[INFO] training model...")

recognizer = create_model()
recognizer.summary()
recognizer.fit(np.array(data['embeddings']),labels,epochs=20)

recognizer.save("output/model")

f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()