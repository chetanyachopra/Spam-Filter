import pandas as pd 
import pprint
import numpy as np


#make a dataframe
df = pd.read_table('dataset/SMSSpamCollection', sep ='\t',  header =None, names = ['label', 'message'])

test_string = "Congratulations you won lottery of 100000$ to aavail give your bank info"

#label features with 0 and 1
df['label'] = df.label.map({'ham':0, 'spam':1})

#dividing in training and testing data
from sklearn.model_selection import train_test_split as tts

x_message_train, x_message_test, y_label_train, y_label_test = tts(df['message'], df['label'], random_state = 1)
#implimenting bag of words
# pprint.pprint(type(x_message_train))

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words = 'english')
#print(count_vector)
training_data = count_vector.fit_transform(x_message_train)
testing_data = count_vector.transform(x_message_test)
#testStringVector = count_vector.
#print(testing_data)

import pickle

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(training_data, y_label_train)
predictions = nb.predict(testing_data)
#print(predictions)

saved_classifier_file = open('my_c.pickle', 'wb')
pickle.dump(nb, saved_classifier_file)


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
print("Accuracy Score: "+format(accuracy_score(y_label_test, predictions)))
print("Precision Score: "+ format(precision_score(y_label_test, predictions)))
print("Recall Score : "+format(recall_score(y_label_test, predictions)))
print("f1 Score : "+format(f1_score(y_label_test, predictions)))
