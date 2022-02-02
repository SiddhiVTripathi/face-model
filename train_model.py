# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import argparse
import pickle
import os
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

print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
# output/embeddings.pickle 
print("[INFO] training model...")
recognizer =  MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
recognizer.fit(data["embeddings"], labels)

f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()
# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
