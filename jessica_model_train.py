##############################

from jessica_deep_emotion_sensor import *
from jessica_emoint_data_conversion import *

fear_texts, fear_tags = convert_file_to_text_and_tag_list(
	emotion_tag = "fear",
	data_file = "*.train.txt")
fear_tagger = train_tagger(texts = fear_texts,
	tags = fear_tags,
	tagger_model_path = 'fear_tagger.h5',
	epochs = 10)
'''
Epoch 10/10
3251/3251 [==============================] - 6s 2ms/sample - loss: 0.0115 - acc: 0.9917 - val_loss: 0.3208 - val_acc: 0.9006
'''

anger_texts, anger_tags = convert_file_to_text_and_tag_list(
	emotion_tag = "anger",
	data_file = "*.train.txt")
anger_tagger = train_tagger(texts = anger_texts,
	tags = anger_tags,
	tagger_model_path = 'anger_tagger.h5',
	epochs = 10)
'''
Epoch 10/10
3251/3251 [==============================] - 6s 2ms/sample - loss: 0.0131 - acc: 0.9905 - val_loss: 0.2664 - val_acc: 0.9199
'''

joy_texts, joy_tags = convert_file_to_text_and_tag_list(
	emotion_tag = "joy",
	data_file = "*.train.txt")
joy_tagger = train_tagger(texts = joy_texts,
	tags = joy_tags,
	tagger_model_path = 'joy_tagger.h5',
	epochs = 10)
'''
Epoch 10/10
3251/3251 [==============================] - 7s 2ms/sample - loss: 5.7410e-05 - acc: 1.0000 - val_loss: 6.1900 - val_acc: 0.3425
'''

sadness_texts, sadness_tags = convert_file_to_text_and_tag_list(
	emotion_tag = "sadness",
	data_file = "*.train.txt")
sadness_tagger = train_tagger(texts = sadness_texts,
	tags = sadness_tags,
	tagger_model_path = 'sadness_tagger.h5',
	epochs = 10)
'''
Epoch 10/10
3251/3251 [==============================] - 6s 2ms/sample - loss: 0.0212 - acc: 0.9852 - val_loss: 0.2466 - val_acc: 0.9309
'''

'''
cp *.h5 /Downloads/
'''
##############################