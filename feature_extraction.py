


#--
# Length of text, number of words, average length of words, number of short words, , 
# , , , + frequency of 12 punctuation marks.  
# 
# Hybrid features- frequencies of 100 most frequent character-level bi-grams and ri-grams. 

from string import punctuation
from collections import Counter
from itertools import chain
import re


class documentFeatures():

    # punctuation = ".?\"',-!:;([/"
    # punctuation2 = ".?\"',-!:;)]/" 

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
        
        with open(path, 'r') as f:
            for line in f:
                for word in line.strip(punctuation).split():

                    self.word_number+=1


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

        
        self.calculate(path)

    



jo = documentFeatures("ac1.txt")
jo2 = documentFeatures("pg1.txt")
jo3 = documentFeatures("pablo1.txt")

print(len(jo.hapax_legomena))
print(len(jo.dislegomenon))
print(len(jo.hapax_legomena)/jo.word_number)
print(len(jo.dislegomenon)/jo.word_number)

print(len(jo2.hapax_legomena))
print(len(jo2.dislegomenon))
print(len(jo2.hapax_legomena)/jo2.word_number)
print(len(jo2.dislegomenon)/jo2.word_number)


print(len(jo3.hapax_legomena))
print(len(jo3.dislegomenon))
print(len(jo3.hapax_legomena)/jo2.word_number)
print(len(jo3.dislegomenon)/jo2.word_number)