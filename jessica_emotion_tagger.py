########jessica_emotion_tagger.py########
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import *

fear_tagger = keras.models.load_model('fear_tagger.h5')
sadness_tagger = keras.models.load_model('sadness_tagger.h5')
joy_tagger = keras.models.load_model('joy_tagger.h5')
anger_tagger = keras.models.load_model('anger_tagger.h5')

tagger_models = {"fear": fear_tagger,
"sadness": sadness_tagger,
"joy": joy_tagger,
"anger": anger_tagger}

def emotion_tagging(text):
	x = texts_to_input([text])
	output_tags = []
	for emotion in tagger_models:
		emotion_tagger = tagger_models[emotion]
		scores = emotion_tagger.predict(x)
		prediction = np.argmax(scores)
		confidence = np.max(scores)
		if prediction == 1:
			output_tags.append({'tag':emotion, "confidence":confidence})
	return output_tags

'''

text = "I feel so afraid ! #fear"

text = "A pessimist is someone who, when opportunity knocks, complains about the noise #mikeshumor"

text = "Is it me, or is Ding wearing the look of a man who's just found his arch enemy in bed with his missus? #angryman "

text = "Be it a rainy day, be it cheerful sunshine, I am a Prussian, want nothing to be but a Prussian.:|"

emotion_tagging(text)
'''
########jessica_emotion_tagger.py########