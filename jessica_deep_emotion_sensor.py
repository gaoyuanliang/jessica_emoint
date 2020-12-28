##########jessica_deep_emotion_sensor.py##########

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import *

# Model constants.
max_features = 20000
embedding_dim = 300
sequence_length = 100

def texts_to_input(texts):
	word_id_sequence = map(lambda x: tf.keras.preprocessing.text.one_hot(x, n=max_features), 
		texts)
	word_id_sequence = list(word_id_sequence)
	x = np.array(word_id_sequence)
	x = tf.keras.preprocessing.sequence.pad_sequences(
		x, padding="post",
		maxlen=sequence_length,
	)
	return x

def emotion_tagger_model_building(
	max_features = 20000,
	embedding_dim = 300,
	sequence_length = 100):
	# A integer input for vocab indices.
	inputs = tf.keras.Input(shape=(sequence_length), dtype="int64")
	# Next, we add a layer to map those vocab indices into a space of dimensionality
	# 'embedding_dim'.
	x = layers.Embedding(max_features, embedding_dim)(inputs)
	x = layers.Dropout(0.5)(x)
	# Conv1D + global max pooling
	x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
	x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
	x = layers.GlobalMaxPooling1D()(x)
	# We add a vanilla hidden layer:
	x = layers.Dense(128, activation="relu")(x)
	x = layers.Dropout(0.5)(x)
	# We project onto a single unit output layer, and squash it with a sigmoid:
	predictions = layers.Dense(2, activation="softmax", name="predictions")(x)
	model = tf.keras.Model(inputs, predictions)
	model.compile(
		loss="categorical_crossentropy", 
		optimizer="adam", 
		metrics=["accuracy"])
	return model

def train_tagger(texts,
	tags,
	tagger_model_path,
	epochs = 100,
	):
	tagger_model = emotion_tagger_model_building()
	'''
	prepare the text input

	texts = [
		"i feel so fear",
		"nothing is wrong"
		]
	'''
	x = texts_to_input(texts)
	'''
	prepare the output
	'''
	y = np.array(tags)
	y = to_categorical(y)
	print(x.shape, y.shape)
	# Fit the model using the train and test datasets.
	tagger_model.fit(x, y, 
		validation_split=0.1, 
		epochs=epochs)
	tagger_model.save(tagger_model_path,
		save_format='h5')
	return tagger_model

##########jessica_deep_emotion_sensor.py##########