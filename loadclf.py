import pickle

f = open('my_c.pickle', 'rb')
clf = pickle.load(f)
print(clf)

test_string = "Congratulations you won lottery of 100000$ to aavail give your bank info"

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(stop_words = 'english')
test_data = count_vector.transform(test_string)
pre = clf.predict(test_data)

f.close()