__author__ = 'Noblesse Oblige'
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation

from FakeNews.baseline.fnc_kfold import *
from FakeNews.baseline.feature_engineering import *
from FakeNews.utils.dataset import *
from FakeNews.utils.generate_test_splits import *
from FakeNews.utils.score import *
from FakeNews.utils.system import *




check_version()
parse_params()

#Load the training dataset and generate folds
d = DataSet(path="fnc_1")
folds,hold_out = kfold_split(d,n_folds=10)
fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

# Load the competition dataset
competition_dataset = DataSet("competition_test")
X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

Xs = dict()
ys = dict()

# Load/Precompute all features now
X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
for fold in fold_stances:
    Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


#Deciding which ML
#
# en=ExtraTreesClassifier(n_estimators=200,bootstrap=True,oob_score=True,n_jobs = -1, random_state=14128)
# rf=RandomForestClassifier(n_estimators=200,random_state=None,n_jobs = -1,bootstrap=True,oob_score=True)
# kn1=KNeighborsClassifier(n_neighbors=3,weights='distance')
# kn2=KNeighborsClassifier()
# nb=make_pipeline(StandardScaler(),GaussianNB())
# svm1=LinearSVC(random_state=None,multi_class='crammer_singer')
# svm2=LinearSVC(random_state=14128)
# lr1=LogisticRegression(random_state=14128,multi_class='multinomial',solver='newton-cg')
# lr2=LogisticRegression(penalty='l1')
# sdg=SGDClassifier(tol=1e-4,loss='perceptron', random_state=14128,n_jobs= -1,penalty='elasticnet')
# nn=MLPClassifier(activation='tanh',solver='lbfgs',learning_rate='adaptive',random_state=14128)
gdb = GradientBoostingClassifier(n_estimators=200, random_state=None, verbose=True)
# param=[rf,en,svm1,sdg,nb]
# # for each system


#print("AAAAAAAAA",score_defaults(y_competition))
p=30
#sub_score=0
#for clf in param:
clf=gdb #joblib.load('fold'+str(31)+'.pkl')
if True:
    p=p+1
  # Classifier for each fold
    best_score = 0
    best_fold = None
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))
#
        X_test = Xs[fold]
        y_test = ys[fold]
        print("test",score_defaults(y_test))

        # Semi Automated: could not be run as system would feeze
        #
        # clf = LabelPropagation()
        # rng = np.random.RandomState(42)
        # random_unlabeled_points = rng.rand(len(y_train)) < 0.3
        # labels = np.copy(y_train)
        # labels[random_unlabeled_points] = -1
        # clf.fit(X_train, labels)
#
#
#finding best params
        # GBCparams={'n_estimators':[100,200,300],'min_samples_split':[2,3], 'random_state':[None,14128]}
        # gbc = GradientBoostingClassifier()
        # clf = GridSearchCV(gbc, GBCparams)
#
#         # SVMparams={'multi_class':['ovr','crammer_singer'],'fit_intercept':[True,False],'random_state':[None,14128]}
#         # svm=LinearSVC()
#         # clf = GridSearchCV(svm, SVMparams)
#         #
#         # LRparams={'penalty':['l1','l2'],'dual':[False,True], 'fit_intercept':[True,False],'class_weight':['balanced',None],'random_state':[None,14128], 'solver':['newton-cg','lbfgs','liblinear','saga'] ,'multi_class':['ovr','multinomial'], }
#         # lr=LogisticRegression()
#         # clf = GridSearchCV(lr, LRparams)
#         #
        #SDGparams={'loss':['hinge','log','modified_huber','squared_hinge','perceptron','huber','squared_loss'],'penalty':[None,'l2','l1','elasticnet'], 'fit_intercept':[True,False],'random_state':[None,14128],'learning_rate':['constant','optimal','invscaling'],'tol':[1e-3,1e-4,1e-5],'eta0':[0.1,0.5,1.0],'average':[True,False]}
        #sdg=SGDClassifier()
        #clf = GridSearchCV(sdg, SDGparams)
#         #
#         NNparams={'activation':['logistic','tanh','relu'],'solver':['lbfgs','sgd'],'learning_rate':['constant','invscaling','adaptive'],'random_state':[None,14128]}
#         nn=MLPClassifier()
#         clf = GridSearchCV(nn, NNparams)
#         #
#         # NBparams={}
#         # nb=GaussianNB()
#         # clf = GridSearchCV(nb, NBparams)
#         #
#         # KNparams={'n_neighbors':[3,4,5,7],'weights':['uniform','distance'],'leaf_size':[30,10,50]}
#         # kn=KNeighborsClassifier()
#         # clf = GridSearchCV(kn, KNparams)
#         #
#         # RFparams={'n_estimators':[10,100],'max_features':['auto','log2',None],'oob_score':[False,True],'random_state':[None,14128]}
#         # rf=RandomForestClassifier()
#         # clf = GridSearchCV(rf, RFparams)
#
        clf.fit(X_train, y_train)
        print(fold,' ssh ',p)
#
        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]
        #
        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)
        #
        score = fold_score/max_fold_score
        #print("SCORE::",score)
        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf
            joblib.dump(clf, 'fold'+str(p)+'.pkl')

    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]
    print("Dev",score_defaults(actual))
    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]
    print("Comp",score_defaults(actual))

    print("Scores on the test set")
    report_score(actual,predicted)
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
#
for i in range(1,26):
     try:
        best_fold = joblib.load('fold'+str(i)+'.pkl')

        #Run on Holdout set and report the final score on the holdout set
        predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
        actual = [LABELS[int(a)] for a in y_holdout]
        print("Dev",i,score_defaults(actual))
        print("Scores on the dev set")
        report_score(actual,predicted)
        print("")
        print("")

        #Run on competition dataset
        predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
        actual = [LABELS[int(a)] for a in y_competition]
        print("Comp",i,score_defaults(actual))

        print("Scores on the test set")
        report_score(actual,predicted)
#
#
#         Xs[len(folds)]=X_holdout
#         ys[len(folds)]=y_holdout
#         ids = list(range(len(folds)+1))
#         X_train = np.vstack(tuple([Xs[i] for i in ids]))
#         y_train = np.hstack(tuple([ys[i] for i in ids]))
#
#         best_fold.fit(X_train,y_train)
#         #Run on competition dataset
#         predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
#         actual = [LABELS[int(a)] for a in y_competition]
#
#         print("Scores on the test set (full train)")
#         report_score(actual,predicted)
#
     except:
        pass
