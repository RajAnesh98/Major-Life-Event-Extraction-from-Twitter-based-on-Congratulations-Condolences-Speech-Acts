# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 02:58:33 2019

@author: Raj Anesh
"""

# Perform standard imports
import spacy
nlp = spacy.load('en_core_web_sm')

def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')
        
        
f=open(r"C:\Users\Raj Anesh\Desktop\pos.txt")
data =f.read()
x=data.split("@")
for d in x:
    mod = x.index(d) % 2
    if mod > 0:
     print("@@@New Tweett@@@")
     print('\n')
     print(d)
     print('\n')
    #print("@@@New Tweett@@@")
     doc = nlp(d)
     show_ents(doc)
     print('\n')
