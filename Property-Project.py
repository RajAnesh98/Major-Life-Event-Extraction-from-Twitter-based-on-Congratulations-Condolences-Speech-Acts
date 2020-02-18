# -*- coding: utf-8 -*-
"""
Created on Sat May 25 01:04:38 2019

@author: Raj Anesh
"""

import nltk

from nltk.stem.porter import * 
stemmer = PorterStemmer() 
print('studying  --> '+ stemmer.stem('studying'))
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

f=open(r"C:\Users\Raj Anesh\Desktop\pos.txt")
data =f.read()
x=data.split("@")
for d in x:
    print("@@@New Tweett@@@")
    print(d)
    #print("@@@New Tweett@@@")
    doc = nlp(d)
    #displacy.render(doc,style='ent',spyder=True)
    print([(X.text, X.label_) for X in doc.ents])
    print('\n')
    
    
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem.porter import *
stemmer = PorterStemmer()

f= open(r"C:\Users\Raj Anesh\Desktop\pos.txt")
pos_data=f.read()
    
tweets=pos_data.split("@")
stemmed_tweets=[]
for i in tweets:
    mod = tweets.index(i) % 2
    if mod > 0:
        print("ok")
    else:
        a=i.split()
        b=""
        for j in a:
         b=b+" "+stemmer.stem(j)
         stemmed_tweets.append(b)
         cv= CountVectorizer(lowercase=True,stop_words='english')
         dictionary=cv.fit_transform(stemmed_tweets)
         LDA = LatentDirichletAllocation(n_components=1,random_state=10)
         LDA.fit(dictionary)
         for i,topic in enumerate(LDA.components_):
          print(f"The top five words for topic #{i}")
          print([cv.get_feature_names()[index] for index in topic.argsort()[-5:]])
          print('\n')
          print(tweets.index(i))
          print('\n')