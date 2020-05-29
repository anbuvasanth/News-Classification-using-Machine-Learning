import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import numpy as np
import csv
import spacy
nlp=spacy.load('en_core_web_sm')
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from csv import writer

category_list = ["sport", "world", "us", "business", "health", "entertainment", "sci_tech"]


##       ["headline of news","content","url","id","day and date if available"]  
MyList = ["houston astros are sold to local businessman",
          "rapper ja rule pleaded guilty on tuesday to failing to file tax returns and promised to pay more than $1 million in back taxes and penalties,the u.s. attorney's office in new jersey said",
         "http://feeds.reuters.com/~r/",
          "95"]

####                  OR   Give file path      ####

###   but that file text order must ["headline of news","content","url","id","day and date if available"]
##with open('C:\\Users\\Vasanth\\Desktop\\newsclassify\\newsdatas\\entertainment\\12.txt','r')as f:
##     text = f.read()
##     MyList = text.split("\n")



docs_new = MyList

##load the mlpclassifier model
loaded_mpl = pickle.load(open("mlpclassi.pkl","rb"))
predicted=loaded_mpl.predict(docs_new)
fin=category_list [predicted[0]]


##creation of csv file
##with open("C:\\Users\\Vasanth\\Desktop\\newsclassify\\output.csv","w") as f:
##    write=csv.writer(f)
##
##    write.writerow(['ID','Keywords'])

###insert the id and identifier keyword in csv file
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

##Lemmatization
wordnet=WordNetLemmatizer()

###remove stopwords,symbols and numbers from sentences and do lemmatize
corpus=[]
for i in range(len(MyList)):
    rev=re.sub('[^a-zA-Z]',' ',MyList[i])
    rev=rev.lower()
    rev=rev.split()
    rev=[wordnet.lemmatize(word) for word in rev if not word in set(stopwords.words('english'))]
    rev=' '.join(rev)
    corpus.append(rev)

###Named Entity Recognition     
result=[]
for i in range(len(corpus)):
    docx=nlp(corpus[i])
    [result.append(entity.text) for entity in docx.ents]

###TFIDF VECTORIZER
cv=TfidfVectorizer()
X=cv.fit_transform(result)
final_output=cv.get_feature_names()

final_output='->'.join(final_output)
idno=MyList[3]

row_contents=[idno,str(fin+' news=>>'+final_output)]
print(row_contents)

##Pass the parameter's(id and identifer) to function(append_list_as_row)
append_list_as_row('C:\\Users\\Vasanth\\Desktop\\newsclassify\\output.csv', row_contents)    

