import numpy as np
from scipy import interp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)


data = pd.read_csv('sample_selected.csv',index_col = 0)

feature_V1 = data[['Speaker','word','label','vowel', 'stress', 'dur','pre_seg','fol_seg','pre_word','fol_word',
                          'F1', 'F2','F3', 'B1', 'B2', 'B3',
                          'F1@20%', 'F2@20%','F1@35%', 'F2@35%', 'F1@50%', 'F2@50%', 'F1@65%', 'F2@65%', 'F1@80%','F2@80%','nFormants']]
# test vowel by word model 
#feature_V1 = data_featureV1[data_featureV1['word'] == 'AMENDMENT']
feature_V1 = feature_V1.dropna()

vowel_encoder = preprocessing.LabelEncoder()
vowel_encoder.fit(feature_V1.vowel)
feature_V1['vowel_num'] = vowel_encoder.transform(feature_V1.vowel) 

word_encoder = preprocessing.LabelEncoder()
word_encoder.fit(feature_V1.word)
feature_V1['word_num'] = word_encoder.transform(feature_V1.word) 

pre_seg_encoder = preprocessing.LabelEncoder()
pre_seg_encoder.fit(feature_V1.pre_seg)
feature_V1['pre_seg_num'] = pre_seg_encoder.transform(feature_V1.pre_seg) 

fol_seg_encoder = preprocessing.LabelEncoder()
fol_seg_encoder.fit(feature_V1.fol_seg)
feature_V1['fol_seg_num'] = fol_seg_encoder.transform(feature_V1.fol_seg)

pre_word_encoder = preprocessing.LabelEncoder()
pre_word_encoder.fit(feature_V1.pre_word)
feature_V1['pre_word_num'] = pre_word_encoder.transform(feature_V1.pre_word) 

fol_word_encoder = preprocessing.LabelEncoder()
fol_word_encoder.fit(feature_V1.fol_word)
feature_V1['fol_word_num'] = fol_word_encoder.transform(feature_V1.fol_word)

feature_V1.index = feature_V1['Speaker']

label_speaker = feature_V1.groupby(['Speaker'])['label'].mean()
speaker = np.array(list(label_speaker.index))


X, y = feature_V1.drop(['Speaker','label','vowel','word','pre_seg','fol_seg', 'pre_word', 'fol_word'],axis=1), feature_V1['label']
def training_model(X, y, classifier, fig_name):
	speaker_pred = dict.fromkeys(list(pd.Series(y.index).unique()))
	for key in speaker_pred.keys():
		speaker_pred[key]=[]

	cv = StratifiedKFold(n_splits=10)
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)

	colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange']
	lw = 2
	i = 0
	plt.figure(figsize = (10,10))
	accuracy_all = []
	for (train, test), color in zip(cv.split(speaker,label_speaker), colors):
	    train_s = speaker[train]
	    test_s = speaker[test]
	    clf = classifier.fit(X.ix[train_s], y.ix[train_s])
	    probas_,label = clf.predict_proba(X.ix[test_s]),clf.predict(X.ix[test_s])
	    # Compute ROC curve and area the curve
	    acc = []
	    prediction_byspeaker = pd.DataFrame(y.ix[test_s] == label )
	    prediction_byspeaker['Speaker'] = list(prediction_byspeaker .index.get_level_values('Speaker'))
	    prediction_byspeaker = prediction_byspeaker.groupby(['Speaker','label'])['label'].count()
	    for s in prediction_byspeaker.index.get_level_values('Speaker').unique():
	        p = prediction_byspeaker.ix[s].idxmax()
	        acc.append(p)
	        speaker_pred[s].append(p)
	    accuracy = sum(acc)/len(acc)
	    accuracy_all.append(accuracy)
	    fpr, tpr, thresholds = roc_curve(y.ix[test_s], probas_[:, 1])
	    roc_auc = auc(fpr, tpr)
	    if roc_auc < 0.5:
	        fpr, tpr, thresholds = roc_curve(y.ix[test_s], 1-probas_[:, 1])
	        roc_auc = auc(fpr, tpr)
	    mean_tpr += interp(mean_fpr, fpr, tpr)
	    mean_tpr[0] = 0.0
	    plt.plot(fpr, tpr, lw=lw, color=color,
	             label='ROC fold %d (area = %0.2f) (Accuracy by Speaker = %0.2f)' % (i, roc_auc, accuracy))

	    i += 1
	    
	plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
	         label='Luck')

	mean_tpr /= cv.get_n_splits(speaker,label_speaker)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	#speaker_pred = pd.DataFrame(speaker_pred).T

	plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
	         label='Mean ROC (area = %0.2f) (Accuracy by Speaker = %0.2f) ' % (mean_auc,sum(accuracy_all)/len(accuracy_all)), lw=lw)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.savefig(fig_name+'.png')



X, y = feature_V1.drop(['Speaker','label','vowel','word','pre_seg','fol_seg', 'pre_word', 'fol_word'],axis=1), feature_V1['label']
# Classification Tree max_depth 10
#classifier = tree.DecisionTreeClassifier(max_depth=10)
#training_model(X, y, classifier, 'ctree_10')

# Classification Tree max_depth 16
#classifier = tree.DecisionTreeClassifier(max_depth=16)
#training_model(X, y, classifier, 'ctree_16')

# Classification Tree max_depth 6
#classifier = tree.DecisionTreeClassifier(max_depth=6)
#training_model(X, y, classifier, 'ctree_6')

# Gradient Boosting 50
#classifier = GradientBoostingClassifier(n_estimators=50)
#training_model(X, y, classifier, 'GBoost_50')
# Gradient Boosting 100
#classifier = GradientBoostingClassifier(n_estimators=100)
#training_model(X, y, classifier, 'GBoost_100')


#SVM
X_n = pd.DataFrame(preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(X,1),
                     index = X.index,columns = X.columns)
classifier = svm.SVC()
training_model(X_n, y, classifier,'SVM')
# Random Forest
classifier = RandomForestClassifier(max_depth=16, n_estimators=50)
training_model(X, y, classifier, 'RandomForest')

# speaker_pred = dict.fromkeys(list(pd.Series(y.index).unique()))
# for key in speaker_pred.keys():
#     speaker_pred[key]=[]
# cv = StratifiedKFold(n_splits=10)
# classifier = tree.DecisionTreeClassifier(max_depth=10)#, min_samples_leaf=300)
# ean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)

# colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
# lw = 2

# i = 0
# plt.figure(figsize = (10,10))

# for (train, test), color in zip(cv.split(speaker,label_speaker), colors):
#     train_s = speaker[train]
#     test_s = speaker[test]
#     clf = classifier.fit(X.ix[train_s], y.ix[train_s])
#     probas_,label = clf.predict_proba(X.ix[test_s]),clf.predict(X.ix[test_s])
#     # Compute ROC curve and area the curve
#     acc = []
#     prediction_byspeaker = pd.DataFrame(y.ix[test_s] == label )
#     prediction_byspeaker['Speaker'] = list(prediction_byspeaker .index.get_level_values('Speaker'))
#     prediction_byspeaker = prediction_byspeaker.groupby(['Speaker','label'])['label'].count()
#     for s in prediction_byspeaker.index.get_level_values('Speaker').unique():
#         p = prediction_byspeaker.ix[s].idxmax()
#         acc.append(p)
#         speaker_pred[s].append(p)
#     accuracy = sum(acc)/len(acc)
    
#     fpr, tpr, thresholds = roc_curve(y.ix[test_s], probas_[:, 1])
#     roc_auc = auc(fpr, tpr)
#     if roc_auc < 0.5:
#         fpr, tpr, thresholds = roc_curve(y.ix[test_s], 1-probas_[:, 1])
#         roc_auc = auc(fpr, tpr)
#     mean_tpr += interp(mean_fpr, fpr, tpr)
#     mean_tpr[0] = 0.0
#     plt.plot(fpr, tpr, lw=lw, color=color,
#              label='ROC fold %d (area = %0.2f) (Accuracy by Speaker = %0.2f)' % (i, roc_auc, accuracy))

#     i += 1
    
# plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
#          label='Luck')

# mean_tpr /= cv.get_n_splits(speaker,label_speaker)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# speaker_pred = pd.DataFrame(speaker_pred).T

# plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
#          label='Mean ROC (area = %0.2f) (Accuracy by Speaker = %0.2f) ' % (mean_auc,speaker_pred.sum()[0]/len(speaker_pred)), lw=lw)

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.savefig('ctree_10.png')

# Classification Tree max_depth 16
# X, y = feature_V1.drop(['Speaker','label','vowel','word','pre_seg','fol_seg', 'pre_word', 'fol_word'],axis=1), feature_V1['label']

# speaker_pred = dict.fromkeys(list(pd.Series(y.index).unique()))
# for key in speaker_pred.keys():
#     speaker_pred[key]=[]
# cv = StratifiedKFold(n_splits=10)
# classifier = tree.DecisionTreeClassifier(max_depth=16)#, min_samples_leaf=300)
# ean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)

# colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
# lw = 2

# i = 0
# plt.figure(figsize = (10,10))

# for (train, test), color in zip(cv.split(speaker,label_speaker), colors):
#     train_s = speaker[train]
#     test_s = speaker[test]
#     clf = classifier.fit(X.ix[train_s], y.ix[train_s])
#     probas_,label = clf.predict_proba(X.ix[test_s]),clf.predict(X.ix[test_s])
#     # Compute ROC curve and area the curve
#     acc = []
#     prediction_byspeaker = pd.DataFrame(y.ix[test_s] == label )
#     prediction_byspeaker['Speaker'] = list(prediction_byspeaker .index.get_level_values('Speaker'))
#     prediction_byspeaker = prediction_byspeaker.groupby(['Speaker','label'])['label'].count()
#     for s in prediction_byspeaker.index.get_level_values('Speaker').unique():
#         p = prediction_byspeaker.ix[s].idxmax()
#         acc.append(p)
#         speaker_pred[s].append(p)
#     accuracy = sum(acc)/len(acc)
    
#     fpr, tpr, thresholds = roc_curve(y.ix[test_s], probas_[:, 1])
#     roc_auc = auc(fpr, tpr)
#     if roc_auc < 0.5:
#         fpr, tpr, thresholds = roc_curve(y.ix[test_s], 1-probas_[:, 1])
#         roc_auc = auc(fpr, tpr)
#     mean_tpr += interp(mean_fpr, fpr, tpr)
#     mean_tpr[0] = 0.0
#     plt.plot(fpr, tpr, lw=lw, color=color,
#              label='ROC fold %d (area = %0.2f) (Accuracy by Speaker = %0.2f)' % (i, roc_auc, accuracy))

#     i += 1
    
# plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
#          label='Luck')

# mean_tpr /= cv.get_n_splits(speaker,label_speaker)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# speaker_pred = pd.DataFrame(speaker_pred).T

# plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
#          label='Mean ROC (area = %0.2f) (Accuracy by Speaker = %0.2f) ' % (mean_auc,speaker_pred.sum()[0]/len(speaker_pred)), lw=lw)

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.savefig('ctree_16.png')


# # Classification Tree max_depth 6
# X, y = feature_V1.drop(['Speaker','label','vowel','word','pre_seg','fol_seg', 'pre_word', 'fol_word'],axis=1), feature_V1['label']

# speaker_pred = dict.fromkeys(list(pd.Series(y.index).unique()))
# for key in speaker_pred.keys():
#     speaker_pred[key]=[]
# cv = StratifiedKFold(n_splits=10)
# classifier = tree.DecisionTreeClassifier(max_depth=6)#, min_samples_leaf=300)
# ean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)

# colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
# lw = 2

# i = 0
# plt.figure(figsize = (10,10))

# for (train, test), color in zip(cv.split(speaker,label_speaker), colors):
#     train_s = speaker[train]
#     test_s = speaker[test]
#     clf = classifier.fit(X.ix[train_s], y.ix[train_s])
#     probas_,label = clf.predict_proba(X.ix[test_s]),clf.predict(X.ix[test_s])
#     # Compute ROC curve and area the curve
#     acc = []
#     prediction_byspeaker = pd.DataFrame(y.ix[test_s] == label )
#     prediction_byspeaker['Speaker'] = list(prediction_byspeaker .index.get_level_values('Speaker'))
#     prediction_byspeaker = prediction_byspeaker.groupby(['Speaker','label'])['label'].count()
#     for s in prediction_byspeaker.index.get_level_values('Speaker').unique():
#         p = prediction_byspeaker.ix[s].idxmax()
#         acc.append(p)
#         speaker_pred[s].append(p)
#     accuracy = sum(acc)/len(acc)
    
#     fpr, tpr, thresholds = roc_curve(y.ix[test_s], probas_[:, 1])
#     roc_auc = auc(fpr, tpr)
#     if roc_auc < 0.5:
#         fpr, tpr, thresholds = roc_curve(y.ix[test_s], 1-probas_[:, 1])
#         roc_auc = auc(fpr, tpr)
#     mean_tpr += interp(mean_fpr, fpr, tpr)
#     mean_tpr[0] = 0.0
#     plt.plot(fpr, tpr, lw=lw, color=color,
#              label='ROC fold %d (area = %0.2f) (Accuracy by Speaker = %0.2f)' % (i, roc_auc, accuracy))

#     i += 1
    
# plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
#          label='Luck')

# mean_tpr /= cv.get_n_splits(speaker,label_speaker)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# speaker_pred = pd.DataFrame(speaker_pred).T

# plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
#          label='Mean ROC (area = %0.2f) (Accuracy by Speaker = %0.2f) ' % (mean_auc,speaker_pred.sum()[0]/len(speaker_pred)), lw=lw)

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.savefig('ctree_6.png')



# # Gradient Boosting 50
# X, y = feature_V1.drop(['Speaker','label','vowel','word','pre_seg','fol_seg', 'pre_word', 'fol_word'],axis=1), feature_V1['label']

# speaker_pred = dict.fromkeys(list(pd.Series(y.index).unique()))
# for key in speaker_pred.keys():
#     speaker_pred[key]=[]
# cv = StratifiedKFold(n_splits=10)
# classifier = GradientBoostingClassifier(n_estimators=50)
# ean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)

# colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
# lw = 2

# i = 0
# plt.figure(figsize = (10,10))

# for (train, test), color in zip(cv.split(speaker,label_speaker), colors):
#     train_s = speaker[train]
#     test_s = speaker[test]
#     clf = classifier.fit(X.ix[train_s], y.ix[train_s])
#     probas_,label = clf.predict_proba(X.ix[test_s]),clf.predict(X.ix[test_s])
#     # Compute ROC curve and area the curve
#     acc = []
#     prediction_byspeaker = pd.DataFrame(y.ix[test_s] == label )
#     prediction_byspeaker['Speaker'] = list(prediction_byspeaker .index.get_level_values('Speaker'))
#     prediction_byspeaker = prediction_byspeaker.groupby(['Speaker','label'])['label'].count()
#     for s in prediction_byspeaker.index.get_level_values('Speaker').unique():
#         p = prediction_byspeaker.ix[s].idxmax()
#         acc.append(p)
#         speaker_pred[s].append(p)
#     accuracy = sum(acc)/len(acc)
    
#     fpr, tpr, thresholds = roc_curve(y.ix[test_s], probas_[:, 1])
#     roc_auc = auc(fpr, tpr)
#     if roc_auc < 0.5:
#         fpr, tpr, thresholds = roc_curve(y.ix[test_s], 1-probas_[:, 1])
#         roc_auc = auc(fpr, tpr)
#     mean_tpr += interp(mean_fpr, fpr, tpr)
#     mean_tpr[0] = 0.0
#     plt.plot(fpr, tpr, lw=lw, color=color,
#              label='ROC fold %d (area = %0.2f) (Accuracy by Speaker = %0.2f)' % (i, roc_auc, accuracy))

#     i += 1
    
# plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
#          label='Luck')

# mean_tpr /= cv.get_n_splits(speaker,label_speaker)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# speaker_pred = pd.DataFrame(speaker_pred).T

# plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
#          label='Mean ROC (area = %0.2f) (Accuracy by Speaker = %0.2f) ' % (mean_auc,speaker_pred.sum()[0]/len(speaker_pred)), lw=lw)

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.savefig('GradientBoosting_50.png')

# # Gradient Boosting 50
# X, y = feature_V1.drop(['Speaker','label','vowel','word','pre_seg','fol_seg', 'pre_word', 'fol_word'],axis=1), feature_V1['label']

# speaker_pred = dict.fromkeys(list(pd.Series(y.index).unique()))
# for key in speaker_pred.keys():
#     speaker_pred[key]=[]
# cv = StratifiedKFold(n_splits=10)
# classifier = GradientBoostingClassifier(n_estimators=100)
# ean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)

# colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
# lw = 2

# i = 0
# plt.figure(figsize = (10,10))

# for (train, test), color in zip(cv.split(speaker,label_speaker), colors):
#     train_s = speaker[train]
#     test_s = speaker[test]
#     clf = classifier.fit(X.ix[train_s], y.ix[train_s])
#     probas_,label = clf.predict_proba(X.ix[test_s]),clf.predict(X.ix[test_s])
#     # Compute ROC curve and area the curve
#     acc = []
#     prediction_byspeaker = pd.DataFrame(y.ix[test_s] == label )
#     prediction_byspeaker['Speaker'] = list(prediction_byspeaker .index.get_level_values('Speaker'))
#     prediction_byspeaker = prediction_byspeaker.groupby(['Speaker','label'])['label'].count()
#     for s in prediction_byspeaker.index.get_level_values('Speaker').unique():
#         p = prediction_byspeaker.ix[s].idxmax()
#         acc.append(p)
#         speaker_pred[s].append(p)
#     accuracy = sum(acc)/len(acc)
    
#     fpr, tpr, thresholds = roc_curve(y.ix[test_s], probas_[:, 1])
#     roc_auc = auc(fpr, tpr)
#     if roc_auc < 0.5:
#         fpr, tpr, thresholds = roc_curve(y.ix[test_s], 1-probas_[:, 1])
#         roc_auc = auc(fpr, tpr)
#     mean_tpr += interp(mean_fpr, fpr, tpr)
#     mean_tpr[0] = 0.0
#     plt.plot(fpr, tpr, lw=lw, color=color,
#              label='ROC fold %d (area = %0.2f) (Accuracy by Speaker = %0.2f)' % (i, roc_auc, accuracy))

#     i += 1
    
# plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
#          label='Luck')

# mean_tpr /= cv.get_n_splits(speaker,label_speaker)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# speaker_pred = pd.DataFrame(speaker_pred).T

# plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
#          label='Mean ROC (area = %0.2f) (Accuracy by Speaker = %0.2f) ' % (mean_auc,speaker_pred.sum()[0]/len(speaker_pred)), lw=lw)

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.savefig('GradientBoosting_100.png')


# RandomForest

# X, y = feature_V1.drop(['Speaker','label','vowel','word','pre_seg','fol_seg', 'pre_word', 'fol_word'],axis=1), feature_V1['label']

# speaker_pred = dict.fromkeys(list(pd.Series(y.index).unique()))
# for key in speaker_pred.keys():
#     speaker_pred[key]=[]
# cv = StratifiedKFold(n_splits=10)
# classifier = RandomForestClassifier(max_depth=20, n_estimators=50)
# ean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)

# colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
# lw = 2

# i = 0
# plt.figure(figsize = (10,10))

# for (train, test), color in zip(cv.split(speaker,label_speaker), colors):
#     train_s = speaker[train]
#     test_s = speaker[test]
#     clf = classifier.fit(X.ix[train_s], y.ix[train_s])
#     probas_,label = clf.predict_proba(X.ix[test_s]),clf.predict(X.ix[test_s])
#     # Compute ROC curve and area the curve
#     acc = []
#     prediction_byspeaker = pd.DataFrame(y.ix[test_s] == label )
#     prediction_byspeaker['Speaker'] = list(prediction_byspeaker .index.get_level_values('Speaker'))
#     prediction_byspeaker = prediction_byspeaker.groupby(['Speaker','label'])['label'].count()
#     for s in prediction_byspeaker.index.get_level_values('Speaker').unique():
#         p = prediction_byspeaker.ix[s].idxmax()
#         acc.append(p)
#         speaker_pred[s].append(p)
#     accuracy = sum(acc)/len(acc)
    
#     fpr, tpr, thresholds = roc_curve(y.ix[test_s], probas_[:, 1])
#     roc_auc = auc(fpr, tpr)
#     if roc_auc < 0.5:
#         fpr, tpr, thresholds = roc_curve(y.ix[test_s], 1-probas_[:, 1])
#         roc_auc = auc(fpr, tpr)
#     mean_tpr += interp(mean_fpr, fpr, tpr)
#     mean_tpr[0] = 0.0
#     plt.plot(fpr, tpr, lw=lw, color=color,
#              label='ROC fold %d (area = %0.2f) (Accuracy by Speaker = %0.2f)' % (i, roc_auc, accuracy))

#     i += 1
    
# plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
#          label='Luck')

# mean_tpr /= cv.get_n_splits(speaker,label_speaker)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# speaker_pred = pd.DataFrame(speaker_pred).T

# plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
#          label='Mean ROC (area = %0.2f) (Accuracy by Speaker = %0.2f) ' % (mean_auc,speaker_pred.sum()[0]/len(speaker_pred)), lw=lw)

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.savefig('RandomForest_20_50.png')

#SVM 
# X, y = feature_V1.drop(['Speaker','label','vowel','word','pre_seg','fol_seg', 'pre_word', 'fol_word'],axis=1), feature_V1['label']

# X = pd.DataFrame(preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(X,1),
#                      index = X.index,columns = X.columns)

# speaker_pred = dict.fromkeys(list(pd.Series(y.index).unique()))
# for key in speaker_pred.keys():
#     speaker_pred[key]=[]
# cv = StratifiedKFold(n_splits=10)
# classifier = svm.SVC()
# ean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)

# colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
# lw = 2

# i = 0
# plt.figure(figsize = (10,10))

# for (train, test), color in zip(cv.split(speaker,label_speaker), colors):
#     train_s = speaker[train]
#     test_s = speaker[test]
#     clf = classifier.fit(X.ix[train_s], y.ix[train_s])
#     probas_,label = clf.predict_proba(X.ix[test_s]),clf.predict(X.ix[test_s])
#     # Compute ROC curve and area the curve
#     acc = []
#     prediction_byspeaker = pd.DataFrame(y.ix[test_s] == label )
#     prediction_byspeaker['Speaker'] = list(prediction_byspeaker .index.get_level_values('Speaker'))
#     prediction_byspeaker = prediction_byspeaker.groupby(['Speaker','label'])['label'].count()
#     for s in prediction_byspeaker.index.get_level_values('Speaker').unique():
#         p = prediction_byspeaker.ix[s].idxmax()
#         acc.append(p)
#         speaker_pred[s].append(p)
#     accuracy = sum(acc)/len(acc)
    
#     fpr, tpr, thresholds = roc_curve(y.ix[test_s], probas_[:, 1])
#     roc_auc = auc(fpr, tpr)
#     if roc_auc < 0.5:
#         fpr, tpr, thresholds = roc_curve(y.ix[test_s], 1-probas_[:, 1])
#         roc_auc = auc(fpr, tpr)
#     mean_tpr += interp(mean_fpr, fpr, tpr)
#     mean_tpr[0] = 0.0
#     plt.plot(fpr, tpr, lw=lw, color=color,
#              label='ROC fold %d (area = %0.2f) (Accuracy by Speaker = %0.2f)' % (i, roc_auc, accuracy))

#     i += 1
    
# plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
#          label='Luck')

# mean_tpr /= cv.get_n_splits(speaker,label_speaker)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# speaker_pred = pd.DataFrame(speaker_pred).T

# plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
#          label='Mean ROC (area = %0.2f) (Accuracy by Speaker = %0.2f) ' % (mean_auc,speaker_pred.sum()[0]/len(speaker_pred)), lw=lw)

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.savefig('SVM_rbf.png')



























