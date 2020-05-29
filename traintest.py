import pandas
import glob
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import numpy as np


###get all files store in training_data .two columns data and target
category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]
directory_list = ["C:\\Users\\Vasanth\\Desktop\\newsclassify\\newsdatas\\sport\\*.txt", "C:\\Users\\Vasanth\\Desktop\\newsclassify\\newsdatas\\world\\*.txt",
                  "C:\\Users\\Vasanth\\Desktop\\newsclassify\\newsdatas\\us\\*.txt","C:\\Users\\Vasanth\\Desktop\\newsclassify\\newsdatas\\business\\*.txt",
                  "C:\\Users\\Vasanth\\Desktop\\newsclassify\\newsdatas\\health\\*.txt","C:\\Users\\Vasanth\\Desktop\\newsclassify\\newsdatas\\entertainment\\*.txt",
                  "C:\\Users\\Vasanth\\Desktop\\newsclassify\\newsdatas\\sci_tech\\*.txt",]
text_files = list(map(lambda x: glob.glob(x), directory_list))
text_files = [item for sublist in text_files for item in sublist]

training_data = []
for t in text_files:
    f = open(t, 'r')
    f = f.read()
    t = f.split('\n')
    training_data.append({'data' : t[0] + ' ' + t[1] +' '+ t[2], 'target' : category_list.index(t[6])})

###convert to dataframe
training_data = pandas.DataFrame(training_data, columns=['data', 'target'])
training_data.to_csv('train_data.csv')
#print(training_data)


#GET VECTOR COUNT
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(training_data.data)

#SAVE WORD VECTOR
pickle.dump(count_vect.vocabulary_, open("countvector.pkl","wb"))

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#SAVE TF-IDF
pickle.dump(tfidf_transformer, open("tfidf.pkl","wb"))


####Train and test data's
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(training_data.data,training_data.target,test_size=0.25,random_state=42)

#text_clf = Pipeline([('vect', TfidfVectorizer()), 
 #                     ('clf', MultinomialNB()) ])

####Pipeline concept 
text_clf = Pipeline([('vect', TfidfVectorizer()), 
                      ('alf', MLPClassifier(random_state=1, max_iter=300)) ])

# train the model
mlpclassi=text_clf.fit(x_train, y_train)


##store train model to mlpclassi.pkl pickle file
pickle.dump(mlpclassi, open("mlpclassi.pkl","wb"))










