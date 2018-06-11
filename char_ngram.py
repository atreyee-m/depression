# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:26:02 2017

@author: atr
"""

import nltk, re, glob
import numpy as np
from nltk.tokenize import casual_tokenize
from nltk.tokenize.casual import _replace_html_entities
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeRegression
from sklearn import tree

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from nltk.stem.lancaster import LancasterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction import text 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from pandas import DataFrame
from matplotlib import pyplot as plt
import string

lancaster_stemmer = LancasterStemmer()

lemmatizer = WordNetLemmatizer()

EMOTICONS = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      <3                         # heart
    )"""

# URL pattern from John Gruber, modified by Tom Winzig. See
# https://gist.github.com/winzig/8894715

URLS = r"(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:[a-z]{2,13})(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:[a-z]{2,13})\b/?(?!@))))"


def clean_first(doc):
    # make everything lower case 
    words = doc.lower()   
    # Replace  URLs but some URLs becomes "URLURL"
    words = re.sub(URLS,"URL", words)
    # Replace Numbers 
    # (But we get stuff like 'NUM-NUM' 'NUM.NUMNUM' 'NUMNUM' 'NUMNUM-NUMNUM' 'NUMNUMNUM'
    # 'NUMNUMNUMNUM' 'NUMNUMk' 'NUMnd' 'NUMwhy')
    #cleaned = re.sub('\d+', "NUM", words)
    cleaned = re.sub("[^a-zA-Z\s\W]+", "", words) #replace NUM with nothing
   # cleaned = re.sub(r"([\w\d]+\.)([\w\d]+)", r"\1 \2", words)
   # print(cleaned)
    return(cleaned)
   
def lemmatize(doc):   
    lemma_list = []
    wnl = WordNetLemmatizer()
    # cleans the document
    cleaned_doc = clean_first(doc)
    # split into sentences
    sent_text = nltk.sent_tokenize(cleaned_doc)
    for line in sent_text:
    # lemmatize with casual_tokenizer from http://www.nltk.org/_modules/nltk/tokenize/casual.html#TweetTokenizer (we can use something else)
        lemma_list1 = [wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in pos_tag(casual_tokenize(line))]
        lemma_list = lemma_list + lemma_list1
    return(lemma_list)

def stemmer(doc):
    stem_list= []
    l_stemmer = LancasterStemmer()
    cleaned_doc = clean_first(doc)
    sent_text = nltk.sent_tokenize(cleaned_doc)
    for line in sent_text:
        for word in nltk.word_tokenize(line):
            stem_list.append(l_stemmer.stem(word))
    return(stem_list)

def lem_stemmer(doc):
    stem_list=[]
    lemma = lemmatize(doc)
    l_stemmer = LancasterStemmer()
    for word in lemma:
        stem_list.append(l_stemmer.stem(word))
    return(stem_list)
    
def make_combs():
    combs = []
    poslist=[ 'j', 'n', 'r','v']
    #'CC', 'DT', 'EX',  'UH' 'TO' , 'w'  'i',  'm',
    for L in range(0, len(poslist)+1): 
        for subset in itertools.combinations(poslist, L): # get all the possible combination of POSs
            if subset != ():  
                if len(subset) < 9:
                    combs.append(subset) # append in the list "comb"
    return combs
    #print(combs[1])
    
def get_selected_lemma(doc,comb):
   # combs = make_combs()
    lemma_list = []
    wnl = WordNetLemmatizer()        
    # cleans the document
    cleaned_doc = clean_first(doc)
    # split into sentences
    sent_text = nltk.sent_tokenize(cleaned_doc)
    for line in sent_text:
    # lemmatize with casual_tokenizer from http://www.nltk.org/_modules/nltk/tokenize/casual.html#TweetTokenizer (we can use something else)
       # lemma_list1 = [wnl.lemmatize(i,j[0].lower()) if j[0].lower() in ['a','n','v'] else wnl.lemmatize(i) for i,j in pos_tag(casual_tokenize(line))]
        for j in pos_tag(casual_tokenize(line)):
           
            if str(j[1][0].lower()) in comb:
                if j[1][0].lower() in ['j']: #,'n','v']:
                    lemma = wnl.lemmatize(j[0],'a')
                elif j[1][0].lower() in ['n','v']:
                    lemma = wnl.lemmatize(j[0],j[1][0].lower())
                else:
                    lemma = wnl.lemmatize(j[0])
                lemma_list.append(lemma)
                
    return(lemma_list)


# lemmatize and make n-grams
def make_ngrams(text,n):
    ngram_list = []
    text = re.sub(r"([\w\d]+[\.!?]+)([\w\d]+)", r"\1 \2", text)
    sent_text = nltk.sent_tokenize(text) 
    #print(sent_text)
    for sentence in sent_text:
        lemma_list = lemmatize(sentence) 
        #print(lemma_list)
        for i in range(len(lemma_list) - (n - 1)):
            ngram = lemma_list[i:i+n]
            stringngram = " ".join(ngram) #turn the list into string
            ngram_list.append(stringngram) #append the n-gram of words into a list
    return ngram_list

def make_ChNgrams(text, n): 
    ngram_list=[]
    text = re.sub(r"([\w\d]+[\.!?]+)([\w\d]+)", r"\1 \2", text)
    sent_text = nltk.sent_tokenize(text) 
    for sentence in sent_text:
        #sentence = sentence.strip([\n\t])
        new_sent = re.sub("/\s+/S","",sentence) # remove spaces
        for i in range(len(new_sent) - (n-1)):
            ngram=new_sent[i:i+n]
            ngram_list.append(ngram)
    return ngram_list 


# from https://buhrmann.github.io/tfidf-analysis.html


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25): #Xtr =X_train, features =features = vect.get_feature_names()
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 12), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
    
def top_tfidf_feats(row, features, top_n=50):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


X_train=["what is your name ???", "why is this not working?"]

for n in range(2,5):
    print(n)
    if n !=0:
      #  reddit_train = load_files("/media/mh/EF2A-B9DB/Reddit_Data/Train/")
        #X, y = reddit_train.data, reddit_train.target
        # Check the data


       # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

       # print("samples per class: {}".format(np.bincount(y_train)))
       # print("Data: {}".format(np.bincount(y_test)))
        #print(X_train.shape)


        vect = CountVectorizer(analyzer='char', ngram_range = (n, n),encoding='utf-8')
       
        X_train_n = vect.fit_transform(X_train)
        print('X_train.shape:\n{}'.format(X_train_n.shape))
        print(DataFrame(X_train_n.A, columns=vect.get_feature_names()).to_string())
        df = DataFrame(X_train_n.A, columns=vect.get_feature_names()).to_string()
    
        #print(DataFrame(X_train_n.A, columns=vect.get_feature_names()).to_string())
        filename = 'char'+str(n)
        print("filename -->",filename)
        f= open(filename,'w')
        f.write(df)
        f.close()
         
