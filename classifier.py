import pandas as pd 
import pprint
import numpy as np

#make a dataframe
df = pd.read_table('dataset/SMSSpamCollection', sep ='\t',  header =None, names = ['label', 'message'])

#label features with 0 and 1
df['label'] = df.label.map({'ham':0, 'spam':1})

#dividing in training and testing data
from sklearn.model_selection import train_test_split as tts

x_message_train, x_message_test, y_label_train, y_label_test = tts(df['message'], df['label'], random_state = 1)

#implimenting bag of words
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()

training_data = count_vector.fit_transform(x_message_train)
testing_data = count_vector.transform(x_message_test)

print(eval(str(count_vector.get_feature_names())))
