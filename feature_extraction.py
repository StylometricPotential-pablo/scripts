


#--
# Length of text, number of words, average length of words, number of short words, , 
# , , , + frequency of 12 punctuation marks.  
# 
# Hybrid features- frequencies of 100 most frequent character-level bi-grams and ri-grams. 

from string import punctuation
from collections import Counter
from itertools import chain
import re
import nltk
from nltk.corpus import stopwords


import openai
stop_words = set(stopwords.words('english'))


import gensim
from gensim.models import Word2Vec

import numpy as np





class documentFeatures():

    punctuationList = ".?\"',-!:;()[]/"

    # Calculating the relative rarity of words (doesn't scale well with size) 
    def calculate_hapax_legomena(self, path):
        file = open(path)
        list_of_words = re.findall('\w+', file.read().lower())
        freqs = {key: 0 for key in list_of_words}
        for word in list_of_words:
            freqs[word] += 1
        for word in freqs:
            if freqs[word] == 1:
                self.hapax_legomena.append(word)
            if freqs[word] == 2:
                self.dislegomenon.append(word)
    

    def calculate_texts(self, path):
        with open(path) as f:
            contents = f.read()
            
            
        doc = []
        with open(path, 'r') as f:

            for line in f:
                for word in line.strip(punctuation).split():
                    if word not in stop_words:
                        doc.append(word)
                    if (len(word)<4):
                        self.short_word_number+=1
                    if (word in self.uni_grams):
                        self.uni_grams[word]+=1
                    else:
                        self.uni_grams[word]=1
                    self.word_number+=1
        self.uni_grams = dict(sorted(self.uni_grams.items(), key=lambda item: item[1], reverse=True))

        i = 0
        for key in self.uni_grams.keys():
            if i == 100:
                break
            self.uni_gram_frequency.append(self.uni_grams[key])
            i+=1

        input = " ".join(doc)
        bi_grams = {}

        for i in range(len(doc)-1):
            thisGram = doc[i] + doc[i+1]
            thisGram = thisGram.lower()
            if thisGram in bi_grams:
                bi_grams[thisGram]+=1
            else:
                bi_grams[thisGram]=1
        bi_grams = dict(sorted(bi_grams.items(), key=lambda item: item[1], reverse=True))


        self.bi_grams = bi_grams

        i = 0
        for key in self.bi_grams.keys():
            if i == 20:
                break
            self.bi_gram_frequency.append(self.bi_grams[key])
            i+=1
        print(bi_grams) 
        print(self.bi_gram_frequency)
        # print(input)

        response = openai.Embedding.create(
        input= input,
        model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']


        self.embedding = embeddings


    def calculate(self, path):
        self.calculate_texts(path)
        self.calculate_hapax_legomena(path)

            
         

    def __init__(self, path):
        self.text_length = 0        #text length
        self.word_number = 0        #number of words
        self.word_length = 0        #average worth length
        self.short_word_number = 0  #number of short words
        self.digit_cap_prop = 0     #proportion of digits + capital letters
        self.letter_freqs  = 0      #individual letters + digit frequencies
        self.digit_freqs = 0        #frequency of digits
        self.hapax_legomena = []    #hapax-legomena
        self.dislegomenon = []
        self.richness = 0           #measure of text richness
        self.twelve_freq = [0] * 12 #frequency of 12 punctuation marks
        self.uni_grams = {}         #most common words
        self.uni_gram_frequency = []
        self.vector = 0
        self.embedding = 0
        self.bi_grams = {}
        self.bi_gram_frequency = []

        
        self.calculate(path)

    



import pandas as pd



col_names = ['hapax', 'dis', 'words', 'short']

for i in range(120):
    col_names.append(str(i))
col_names.append('label')


feature_cols = ['hapax', 'dis', 'words', 'short']
for i in range(120):
    feature_cols.append(str(i))

# load dataset
pima = pd.read_csv("new_file.csv", header=None, names=col_names)
print(pima.head)

X = pima[feature_cols] # Features
Y = pima.label

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=16)

from sklearn.linear_model import LogisticRegression

# # instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(type(logreg))

print(logreg.predict(X_test))


object = documentFeatures("documents/pablo/pablo1.txt")


dct = {
    'hapax': [len(object.hapax_legomena)], 
    'dis': [len(object.dislegomenon)], 
    'words': [object.word_number], 
    'short': [object.short_word_number]
}
for i in range(100):
    dct[str(i)] = object.uni_gram_frequency[i]
for i in range(20):
    dct[str(100+i)] = object.bi_gram_frequency[i]

# for i in range(20):
#     dct[str(i)] = object.bi_gram_frequency[i]
# Creating pandas dataframe from numpy array
dataset = pd.DataFrame(dct)

print(logreg.predict(dataset))


import json
data = {}
with open('embeddings.json', 'r') as fp:
    data = json.load(fp)


data["2"].pop(0)

from numpy.linalg import norm


def getVal(index, substract):
    index = str(index)
    print(len(data[index]))
    sum1 = 0
    for val in data[index]:
        sum1 += (np.dot(val,object.embedding)/(norm(val)*norm(object.embedding))) * 1/len(data[index])
    return sum1


import matplotlib.pyplot as plt
colors = ['green', 'red', 'black', 'blue', 'blue']

coefficients = logreg.coef_

embs = []

import csv
with open('new_file.csv', newline='') as f:
    reader = csv.reader(f)
    embs = list(reader)

newList = []

for k in range(len(coefficients[0])):
    sum = 0
    for i in range(4):
        sum+=coefficients[i][k]
    sum = sum/4
    newList.append(sum)

print(coefficients)

ooo = 0
for i in range(0,5):
    color = colors[i]
    vals = coefficients[i]
    currentEmbds = embs[i]
    

    joo = 0
    print(str(i)+ " " + str(len(data[str(i)])))
    for lsst in data[str(i)]:
        sum = 0
        for val in lsst:
            sum+=val
        print(sum * -1 -1)

        sumOfAllVector = 0

        for k in range(len(currentEmbds)-10):
            # sumOfAllVector+= float(currentEmbds[k]) * vals[k]
            sumOfAllVector+= float(newList[k]) * float(currentEmbds[joo])



        plt.plot(sumOfAllVector , sum * -1 -1, marker="o", markersize=10, markeredgecolor="red", markerfacecolor=color)
        joo+=1
        ooo+=1
    


    print(i)


# embedks = object.embedding


plt.show()






data = []

import csv


dct = {}

lst = []

def appendValues(document, index):
    newLst = []
    newLst.append(len(document.hapax_legomena)/document.word_number)
    newLst.append(len(document.dislegomenon)/document.word_number)
    newLst.append(document.word_number)
    newLst.append(document.short_word_number)
    newLst.extend(document.uni_gram_frequency)
    newLst.extend(document.bi_gram_frequency)
    newLst.append(index)
    return newLst



for i in range(1,5):
    if i==4:
        continue
    current_docuement = documentFeatures("documents/ac/ac" + str(i) + ".txt")
    data.append(appendValues(current_docuement, 0))
    lst.append(current_docuement.embedding)

dct[0]=lst

import json




lst = []
for i in range(1,7):
    current_docuement = documentFeatures("documents/pg/pg" + str(i) + ".txt")
    data.append(appendValues(current_docuement, 1))
    lst.append(current_docuement.embedding)

dct[1]=lst


lst = []
for i in range(1,6):

    current_docuement = documentFeatures("documents/pablo/pablo" + str(i) + ".txt")
    data.append(appendValues(current_docuement, 2))
    lst.append(current_docuement.embedding)

dct[2]=lst



lst = []
for i in range(1,4):

    current_docuement = documentFeatures("documents/gpt/gpt" + str(i) + ".txt")
    data.append(appendValues(current_docuement, 3))
    lst.append(current_docuement.embedding)

dct[3]=lst
lst = []
current_docuement = documentFeatures("documents/gpt/gpt5.txt")
data.append(appendValues(current_docuement, 4))
lst.append(current_docuement.embedding)
dct[4]=lst


with open('embeddings.json', 'w') as fp:
    json.dump(dct, fp)


import csv


with open("new_file.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(data)







