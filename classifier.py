import pandas as pd 
import pprint
import numpy as np




def readDataSet():
	#make a dataframe
	df = pd.read_table('dataset/SMSSpamCollection', sep ='\t',  header =None, names = ['label', 'message'])
	
	#label features with 0 and 1
	df['label'] = df.label.map({'ham':0, 'spam':1})
	
	return df

#test_string = "ham Congratulations you won lottery of 100000$ to aavail give your bank info"


def getTrainingTestingData(df):
	from sklearn.model_selection import train_test_split as tts

	x_message_train, x_message_test, y_label_train, y_label_test = tts(df['message'], df['label'], random_state = 1)



	from sklearn.feature_extraction.text import CountVectorizer
#implimenting bag of words
	count_vector = CountVectorizer(stop_words = 'english')
	#print(count_vector)

#dividing in training and testing data
	training_data = count_vector.fit_transform(x_message_train)
	testing_data = count_vector.transform(x_message_test)
	
	return {
		"training_data" : training_data, 
		"testing_data" :testing_data, 
		"y_label_train" : y_label_train, 
		"y_label_test" : y_label_test
		}



def make_predictions(train_test_data):
	from sklearn.naive_bayes import MultinomialNB
	nb = MultinomialNB()
	nb.fit(train_test_data["training_data"], train_test_data["y_label_train"])
	predictions = nb.predict(train_test_data["testing_data"])
	return predictions


def print_accuracy(train_test_data, predictions):
	from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
	print("Accuracy Score: "+format(accuracy_score(train_test_data["y_label_test"], predictions)))
	print("Precision Score: "+ format(precision_score(train_test_data["y_label_test"], predictions)))
	print("Recall Score : "+format(recall_score(train_test_data["y_label_test"], predictions)))
	print("f1 Score : "+format(f1_score(train_test_data["y_label_test"], predictions)))





def runScript():
	df = readDataSet()
	train_test_data = getTrainingTestingData(df)
	predictions = make_predictions(train_test_data)
	print_accuracy(train_test_data, predictions)


runScript()