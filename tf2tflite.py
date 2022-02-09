import tensorflow as tf
saved_model_dir = 'output/model'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('output/model.tflite','wb') as f:
    f.write(tflite_model)