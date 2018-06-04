__author__ = 'Noblesse Oblige'

from tqdm import tqdm
import numpy as np
import nltk
from nltk.corpus import cmudict
from nltk.corpus import opinion_lexicon
from nltk.corpus import wordnet as wn
import string
import os
from nltk.parse import stanford
from nltk.util import ngrams
from nltk.util import skipgrams
from nltk.sentiment.util import *
from nltk.sentiment.sentiment_analyzer import *
from FakeNews.fnc_1.scorer import FNCException, LABELS
from FakeNews.utils.score import *
from sklearn.feature_extraction.text import *
from sklearn.externals import joblib
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit

from sklearn.ensemble import GradientBoostingClassifier

import csv


def report_score(actual,predicted):
    score,cm = score_submission(actual,predicted)
    best_score, _ = score_submission(actual,actual)
    print_confusion_matrix(cm)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score


def load_data(filename):
    data = None
    try:
        with open(filename,encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            data = list(reader)

            if data is None:
                error = 'ERROR: No data found in: {}'.format(filename)
                raise FNCException(error)
    except FileNotFoundError:
        error = "ERROR: Could not find file: {}".format(filename)
        raise FNCException(error)

    return data

def CollectData(text):

    stances=load_data("fnc_1/"+text+"_stances.csv")
    body=load_data("fnc_1/"+text+"_bodies.csv")

    stance_body=dict()
    for article in body:
        stance_body[int(article['Body ID'])] = article['articleBody']
    stan=[]
    for s in stances:#merged so as to be able to do stratisfied folding
        s['articleBody']=stance_body[int(s['Body ID'])]
        stan.append(LABELS.index(s["Stance"]))

    return stances,stan

##Collection Class
class Collect_features:

    def __init__(self,text):
        self.bod_text=[]
        self.head_text=[]
        for sent in text:
            self.bod_text.append(sent["articleBody"].lower())
            self.head_text.append(sent["Headline"].lower())

    def FeatureList(self):
        a=[]

        for line in range(len(self.bod_text)):
            for i, (headline, body) in tqdm(enumerate(zip(self.head_text, self.bod_text))):
                b=[]
                bod=self.bod_text[line]
                head=self.head_text[line]
                b.append(sum([1 if (token in bod and token not in ENGLISH_STOP_WORDS) else 0 for token in self.word_token(head)]))
                b.append(sum([1 if token in bod else 0 for token in self.word_token(head)]))
                b.append(sum([ 1 if " ".join(gram) in bod else 0 for gram in self.Word_k_skip_n_gram(0,2,head)]))
                b.append(sum([ 1 if " ".join(gram) in bod else 0 for gram in self.Word_k_skip_n_gram(0,3,head)]))
                b.append(sum([ 1 if " ".join(gram) in bod else 0 for gram in self.Word_k_skip_n_gram(0,4,head)]))
                b.append(sum([ 1 if " ".join(gram) in bod else 0 for gram in self.Word_k_skip_n_gram(0,5,head)]))

                c2=self.Word_k_skip_n_gram(1,2,bod)
                c3=self.Word_k_skip_n_gram(1,3,bod)
                c4=self.Word_k_skip_n_gram(1,4,bod)
                c5=self.Word_k_skip_n_gram(1,5,bod)
                b.append(sum([ 1 if gram in c2 else 0 for gram in self.Word_k_skip_n_gram(1,2,head)]))
                b.append(sum([ 1 if gram in c3 else 0 for gram in self.Word_k_skip_n_gram(1,3,head)]))
                b.append(sum([ 1 if gram in c4 else 0 for gram in self.Word_k_skip_n_gram(1,4,head)]))
                b.append(sum([ 1 if gram in c5 else 0 for gram in self.Word_k_skip_n_gram(1,5,head)]))


                b.append(self.ratio_Overlap_Words(head,bod))
                b.extend([self.Sentiment(head),self.Sentiment(bod)])

                a.append(b)
        return a

    def word_token(self,line):
        return nltk.word_tokenize(line.lower())

    _wnl = nltk.WordNetLemmatizer()
    def normalize_word(self,w,pos):
        pos=self.pos_to_wn(pos)
        return self._wnl.lemmatize(w,pos).lower()

    def token_lemmas(self,s):
        po=nltk.pos_tag(nltk.word_tokenize(s.lower())) #part_Of_Speach tagger
        s=[self.word_token(t) for t in nltk.sent_tokenize(s.lower())] #tokenise by sentece then word
        p=nltk.pos_tag_sents(s)
        return [self.normalize_word(t,pos) for t,pos in po]

    def pos_to_wn(self,tag):#curtasy of bogs answer to https://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word
        if tag in ['JJ', 'JJR', 'JJS']:
            return wn.ADJ
        #elif tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        #    return wn.NOUN
        elif tag in ['RB', 'RBR', 'RBS']:
            return wn.ADV
        elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            return wn.VERB
        return wn.NOUN


    def Word_k_skip_n_gram(self,k,n,words):
        words=self.word_token(words)
        if k==0:
            ret=list(ngrams(words,n))
        else:
            ret=list(skipgrams(words,n,k))
        return ret

    def ratio_Overlap_Words(self,head,body):
        #A normalised calculation of the number of words shared between the headline and body
        no_int=len(set(self.token_lemmas(head)).intersection(set(self.token_lemmas(body))))
        no_un=len(set(self.token_lemmas(head)).union(set(self.token_lemmas(body))))
        return no_int/no_un

    def Sentiment(self,s,plot=False):
        #baisic sentitment analyser courtesy of nltk.sentiment.util.demo_liu_hu_lexicon
        #breifly modified for additional negation calculation and so it gives a return value
        y = []
        token=[self.word_token(t) for t in nltk.sent_tokenize(s.lower())]
        for sent in token:
            neg=0
            for word in sent:
                if NEGATION_RE.search(word):
                    neg+=1
                if word in opinion_lexicon.positive():
                    y.append(1) # positive
                elif word in opinion_lexicon.negative():
                    y.append(-1) # negative
                else:
                    y.append(0) # neutral
            if neg%2 != 0: y.append(-1)
        return sum(y)/len(token)

    def tdif(self,doc_train,doc_test): #incorperated as it was seen the 3rd place of the FNC-1 leader board put much weight behind it
        for i, (headline, body) in tqdm(enumerate(zip(doc_train, doc_test))):
            #TF
            tDoc=TfidfVectorizer(max_df=0.5,decode_error='ignore',ngram_range=(1,5),stop_words='english',use_idf=False)
            D_train=tDoc.fit_transform(doc_train)
            D_test=tDoc.transform(doc_test)
            Dfeature_name=tDoc.get_feature_names()
            ch_best_doc=SelectKBest(chi2,k=len(Dfeature_name)*0.1)
            D_train=ch_best_doc.fit_transform(D_train)
            D_test=ch_best_doc.transform(D_test)
            Dfeature_name=[Dfeature_name[i] for i in ch_best_doc.get_support(indices=True)]

            #TF-IDF
            tiDoc=TfidfVectorizer(max_df=0.5,decode_error='ignore',ngram_range=(1,5),stop_words='english',use_idf=True,norm='l2')
            D_itrain=tiDoc.fit_transform(doc_train)
            D_itest=tiDoc.transform(doc_test)
            Difeature_name=tiDoc.get_feature_names()
            print(Difeature_name)

            return D_train,D_test,D_itrain,D_itest

    def cosSim(self,X,Y):
        return cosine_similarity(X,Y)

def Train_Test(F_train,F_test,name):
        feat_train=Collect_features(F_train)
        feat_test=Collect_features(F_test)

        #collect normal features
        X_train=feat_train.FeatureList()
        X_test=feat_test.FeatureList()
        print("1")
        if not os.path.isfile("features/Train."+name+".npy") and os.path.isfile("features/Test."+name+".npy"):
            #get the TF and TF-IDF of the body and headlines
            BTrain, BTest,BiTrain,BiTest=feat_train.tdif(feat_train.bod_text,feat_test.bod_text)
            HTrain, HTest,HiTrain,HiTest=feat_train.tdif(feat_train.head_text,feat_test.head_text)

            print("2")##Get Cosine similarity of TF-IDF of head and body
            iTest=feat_train.cosSim(HiTest,BiTest)
            iTrain=feat_train.cosSim(HiTrain,BiTrain)

            print("3")##collect together the various metrics
            X_train=np._c[X_train,HTrain,BTrain,iTrain]
            X_test=np._c[X_test,HTest,BTest,iTest]


            np.save("features/Train."+name+".npy", X_train)
            np.save("features/Test."+name+".npy", X_test)

        return np.load("features/Train."+name+".npy"),np.load("features/Test."+name+".npy")

if __name__ == "__main__":
    #Collect Data
    d_data,d_target=CollectData("train")
    d_data=np.array(d_data)
    d_target=np.array(d_target)

    print("A")
    #Collect features from Compitition

    c_data,c_target=CollectData("competition_test")
    c_data=np.array(c_data)
    c_target=np.array(c_target)
    X_data,X_competition=Train_Test(d_data,c_data,"COMPETITION")

    best_score=0
    best_fold=None
    try:
        best_fold=joblib.load('trainedML.pkl')
    except FileNotFoundError:
        ss=StratifiedShuffleSplit(n_splits=3,test_size=0.2,train_size=0.8,random_state=1148925)
        dev=0
        for train,test in ss.split(d_data,d_target):
            hand,hold=d_data[train],d_data[test]
            hand_stances,hold_stances=d_target[train],d_target[test]

            #Collect the Development set features
            dev+=1
            X_hand,X_holdout=Train_Test(hand,hold,"DEVELOPMENT_"+dev)

            # clf.fit(X_hand, hand_stances)
            # joblib.dump(clf, 'trainedML.pkl')
            # print("B")
            # predicted = [LABELS[int(a)] for a in clf.predict(X_holdout)]
            # actual = [LABELS[int(a)] for a in hold_stances]
            # print("Scores on the dev set")
            # report_score(actual,predicted)
            # print("")
            # print("")
            # #Run on competition dataset
            # predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
            # actual = [LABELS[int(a)] for a in c_target]
            # print("Scores on the test set")
            # report_score(actual,predicted)


            sub_score=0
            fold=0
            kf=StratifiedKFold(n_splits=10)
            for train_index, test_index in kf.split(hand,hand_stances):

                F_train,F_test=hand[train_index],hand[test_index]
                y_train,y_test=hand_stances[train_index],hand_stances[test_index]


                #print("C")
                #Collect features for fold
                fold+=1
                X_train,X_test=Train_Test(F_train,F_test,"FOLD_"+fold)
                clf = GradientBoostingClassifier(n_estimators=200, random_state=None, verbose=True)
                clf.fit(X_train, y_train)


                #results of this fold
                predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
                actual = [LABELS[int(a)] for a in y_test]
                fold_score, _ = score_submission(actual, predicted)
                max_fold_score, _ = score_submission(actual, actual)
                score = fold_score/max_fold_score
                print("Score for fold "+ str(test) + " was - " + str(score))
                sub_score +=score


        # The Cross validation analysis
            if best_score<sub_score/10:
                best_score=sub_score/10
                joblib.dump(clf, 'trainedML.pkl')
                best_fold=clf
                print("CHANGED")


        #Run on Dev Set
            predicted = [LABELS[int(a)] for a in clf.predict(X_holdout)]
            actual = [LABELS[int(a)] for a in hold_stances]
            print("Scores on the dev set")
            report_score(actual,predicted)
            print("")
            print("")


    #Run on competition dataset
    best_fold=joblib.load('trainedML.pkl')
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in c_target]
    print("Scores on the test set")
    report_score(actual,predicted)