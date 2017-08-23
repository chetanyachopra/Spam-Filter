import pandas as pd 
import pprint
import numpy as np

df = pd.read_table('dataset/SMSSpamCollection', sep ='\t',  header =None, names = ['label', 'message'])

df['label'] = df.label.map({'ham':0, 'spam':1})

#print(df.shape[0])
# print(df.head())


documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
count_vector.fit(documents)
#pprint.pprint(count_vector.get_feature_names())

doc_arr = count_vector.transform(documents).toarray()
#print(doc_arr)

doc_arr_df = pd.DataFrame(doc_arr, columns = count_vector.get_feature_names())
#print(doc_arr_df)

from sklearn.model_selection import train_test_split as tts

x_message_train, x_message_test, y_label_train, y_label_test = tts(df['message'], df['label'], random_state = 1)

print('total rows = '+format(df.shape[0]))
print('total rows in training = '+ format(x_message_train.shape[0]))
print('total rows in testing = ' + format(x_message_test.shape[0]))

#another change