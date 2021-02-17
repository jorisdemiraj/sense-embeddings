#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
from tqdm import tqdm
import networkx as nx
import numpy as np
import os
from IPython.display import clear_output
from collections import Counter, namedtuple
import multiprocessing
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, FastText
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


# In[25]:


#Below is the code for loading the dataset, parsing, and preprocessing it and in the end saving each sentence on a file.
#The script also does a check for senses included in the mapping, dropping them if not in mapping file. 
import xml.etree.ElementTree as ET
count=0
anncount=0
sentence=[]
annotation=[]
annotations=[]
babelnet=[]
provisore=[]
flag=0
logg=0
v=0

#open the mapping file
with open('bn2wn_mapping.txt', 'r') as fp:
    for line in fp:
        provisore=line.split()
        babelnet.append(provisore[0])
        

#Create a file that will store the sentences
        
with open('corpus.txt', 'wb+') as f:
    #parse through each line in the XML dataset file. The file has been renamed for simplicity of use. 
    for event, elem in ET.iterparse("eurosensehp.xml"):
        #this code checks if the sentences are in english. If true, extract the info needed
        if elem.get('lang')=='en':
 
            if elem.tag=='text':
                count+=1
                print(count,'/1.9 mil')
                clear_output()
               
                if elem.text==None:
                    flag=1
                
                
                else:
                    flag=0
                    strg=elem.text
                    
            if flag==0:
                if elem.tag=='annotation':
               
                    if elem.text not in babelnet:
                        logg+=1
            
                    else:
                        
                        v+=1
                        temp=[]
                        temp=elem.get('lemma').split()
                        temp='_'.join(temp)
                        annotation.extend([[elem.get('anchor'), temp, elem.text]])
        
        if elem.tag=='sentence':
                if v==0:
                    #a check for sentences without annotations is done. If so , drop the sentence
                    annotation=[]
           
                
                else:
                    
                    annotations.append(annotation)
                    for ann in annotations[-1]:
                        strg=strg.replace(ann[0]+' ',ann[1]+'_'+ann[2]+' ')
                    line=(strg+'\n').encode('utf-8')
                    f.write(line)
                    annotation=[]
                    sentence.append(strg)
                    v=0
        elem.clear()
        


# In[2]:


#Loading the sentences from the file and some farther preprocessing for example lowering every uppercase character


# In[79]:


wordList=[]


# In[80]:


count=0
with open('corpus.txt', encoding='utf8') as f:
    
    for entry in f:
        if count==0:
            
            count+=1
        wordList.append(entry.lower().split())


# In[81]:


#remove every word that is not alphanumeric
for i in wordList:
    for j in i:
        if not j[0].isalpha() and not j[0].isdigit(): #<3
            
            
            i.remove(j)
       
   


# In[8]:


#saving the final preprocessed corpus
import pickle
with open("loweredcorpus.pickle", "wb+") as v:
    pickle.dump(wordList, v)


# In[153]:





# In[83]:


#Time to build the model
w2v_model = FastText(min_count=1,workers=cores-1, window=10,size=300)


# In[84]:


#Getting the number of CPU Threads
cores = multiprocessing.cpu_count()


# In[128]:


#Building the vocabolary
w2v_model.build_vocab(wordList)


# In[131]:


#getting the keys from the vocabolary
keys= w2v_model.wv.vocab.keys()


# In[129]:


#time to train the model
w2v_model.train(wordList, total_examples=w2v_model.corpus_count, epochs=50)


# In[17]:


#this function basically extracts only the sense embeddings from the vocabolary, it takes a word as a parameter and it returns the list of senses
def wordcheck(W):
    
    words=[]
    for word in keys:
        temp=word.split(":")
        wordC=temp[0].split("_")
        if temp[-1]=='bn':
            wordC="_".join(temp[:-1])
            if W.lower()==temp.lower():
                words.append(word)
    return words


# In[ ]:


#second method is by getting the senses from wordnet itself


# In[27]:


from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')


# In[46]:


babelnet=[]
wordn=[]
with open('bn2wn_mapping.txt', 'r') as fp:
    for line in fp:
        provisore=line.split()
        babelnet.append(provisore[0])
        wordn.append(provisore[1])


# In[100]:


def wordcheck2(W1):
    lemmas=[]

    offset=[]
    synsets=wn.synsets(W1)
    for syn in synsets:
        offset.append(str(syn.offset()).zfill(8)+'n')
    for sys in synsets:
        lemmas.append(sys.lemma_names())
    newsyn=[]

    countoff=0
    for elem in offset:
        count=0
        countoff+=1
        for elem2 in wordn:
            count+=1
            if elem==elem2:
                for lem in lemmas[countoff-1]:
                    newsyn.append(str(lem)+'_'+str(babelnet[count-1]))
    return newsyn


# In[ ]:


### END OF WORDNET SYNSET EXTRACTION#####


# In[117]:


#The function to check for similarity
def checksimilarity(bank,money):
    x=0
    temp=[]
    if bank==[] or money==[]:
        temp.append(-1)
    else:
        for elem in bank:
            for elem2 in money:
        
                temp.append(w2v_model.wv.similarity(elem, elem2))
                if w2v_model.wv.similarity(elem, elem2)==None:
                    temp.append(-1)
    return temp                   
        


# In[134]:


#This uses the second method of sense extraction
def checksimilarity2(bank,money):
    x=0
    temp=[]
    if bank==[] or money==[]:
        temp.append(-1)
    else:
        for elem in bank:
            for elem2 in money:
                if elem in keys and elem2 in keys:
                    
                    
                    temp.append(w2v_model.wv.similarity(elem, elem2))
                    if w2v_model.wv.similarity(elem, elem2)==None:
                        temp.append(-1)
    return temp                   
        


# In[157]:


#getting the testing data from the file
content=[]

with open('combined.tab') as f:
    linez=f.readlines()
#print(linez)
for elem in linez:
    content.append(elem.replace('\n','').split('\t'))

content.pop(0)


# In[163]:


#time to check for similarity
sim=[]
c=0
for comb in content:
    c+=1
    a=wordcheck(comb[0])
    b=wordcheck(comb[1])
    x=checksimilarity(a,b)
    if x==[]:
        sim.append(-1)
    else:
        sim.append(np.max(x))


# In[164]:


for i in range(len(content)):
    content[i].append(str(sim[i]))


# In[165]:


#append everything to the testing data
a1=[]
a2=[]
for comb in content:
    a1.extend(comb[2].split())
    a2.extend(comb[3].split())


# In[166]:


from scipy.stats import spearmanr


# In[170]:


#Calculate the spearman correlation
print(spearmanr(a1,a2)[0])


# In[23]:


#Save model and weights
w2v_model.wv.save_word2vec_format('embeddings.vec', binary=False)


# In[24]:


w2v_model.save("word2vec.model")


# In[ ]:


#this function is used in case an existing embeddings.vec file is present. 
from gensim.models import KeyedVectors
w2v_model = KeyedVectors.load_word2vec_format('embeddings.vec', binary=False)

