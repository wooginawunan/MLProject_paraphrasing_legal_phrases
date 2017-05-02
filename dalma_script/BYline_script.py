#source activate pythonENVfor_MLProject


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

data_path = 'ML_Project'
data = pd.read_csv(data_path+'/sample_selected.csv',index_col = 0)

feature_V1 = data[['Speaker','word','label','vowel', 'stress', 'dur','pre_seg','fol_seg','pre_word','fol_word',
                          'F1', 'F2','F3', 'B1', 'B2', 'B3',
                          'F1@20%', 'F2@20%','F1@35%', 'F2@35%', 'F1@50%', 'F2@50%', 'F1@65%', 'F2@65%', 'F1@80%','F2@80%']]
# test vowel by word model 
#feature_V1 = data_featureV1[data_featureV1['word'] == 'AMENDMENT']
from sklearn import preprocessing
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
feature_V1 = feature_V1.dropna()

X, y = feature_V1.drop(['Speaker','label','vowel','word','pre_seg','fol_seg', 'pre_word', 'fol_word'],axis=1), feature_V1['label']



# cv = StratifiedKFold(n_splits=6)
# classifier = tree.DecisionTreeClassifier(max_depth=10)#, min_samples_leaf=300)
# mean_tpr = 0.0
# mean_fpr = np.linspace(0, 1, 100)

# colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange']
# lw = 2

# i = 0


# plt.figure(figsize = (10,10))
# for (train, test), color in zip(cv.split(speaker,label_speaker), colors):
#     train_s = speaker[train]
#     test_s = speaker[test]
    
#     probas_ = classifier.fit(X.ix[train_s], y.ix[train_s]).predict_proba(X.ix[test_s])
#     # Compute ROC curve and area the curve
#     fpr, tpr, thresholds = roc_curve(y.ix[test_s], probas_[:, 1])
#     roc_auc = auc(fpr, tpr)
#     if roc_auc < 0.5:
#         fpr, tpr, thresholds = roc_curve(y.ix[test_s], 1-probas_[:, 1])
#         roc_auc = auc(fpr, tpr)
#     mean_tpr += interp(mean_fpr, fpr, tpr)
#     mean_tpr[0] = 0.0
#     plt.plot(fpr, tpr, lw=lw, color=color,
#              label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
#     i += 1

# plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
#          label='Luck')

# mean_tpr /= cv.get_n_splits(speaker,label_speaker)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
#          label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.savefig(data_path+'/ctree_10.png')
#matplotlib.savefig(data_path+'/ctree_10.png')


cv = StratifiedKFold(n_splits=6)
classifier = GradientBoostingClassifier(n_estimators=50)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
plt.figure(figsize = (10,10))
for (train, test), color in zip(cv.split(speaker,label_speaker), colors):
    train_s = speaker[train]
    test_s = speaker[test]
    
    probas_ = classifier.fit(X.ix[train_s], y.ix[train_s]).predict_proba(X.ix[test_s])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y.ix[test_s], probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if roc_auc < 0.5:
        fpr, tpr, thresholds = roc_curve(y.ix[test_s], 1-probas_[:, 1])
        roc_auc = auc(fpr, tpr)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= cv.get_n_splits(speaker,label_speaker)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig(data_path+'/GradientBoosting_50.png')

