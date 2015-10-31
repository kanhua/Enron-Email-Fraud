
# coding: utf-8

# In[274]:

import pandas as pd
import ggplot as gg
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix,classification_report,precision_score,recall_score
from sklearn.preprocessing import scale

# In[275]:

df=pd.read_csv("./data/final_project_dataset.csv")


# In[276]:

ndf=df.drop(["Unnamed: 0","email_address","poi"],axis=1)
#exclude_features=["director_fees","loan_advances","restricted_stock_deferred"]
exclude_features=[]
ndf=ndf.drop(exclude_features,axis=1)
dfmtx=ndf.values
dfmtx.astype(float)
label=df["poi"].values
# Fill in NaN
imp=Imputer(axis=0,strategy="median")
ndfmtx=imp.fit_transform(dfmtx)


# ## Use random forest to select feature

# In[277]:

from sklearn.ensemble import RandomForestClassifier
train_X=ndfmtx
train_y=label
rf=RandomForestClassifier()
rf.fit(train_X,train_y)
rfi=rf.feature_importances_

def list_feature_imp(name,score):
    sorted_rfi_idx=np.argsort(score)
    for r in sorted_rfi_idx:
        print(name[r]+":"+str(score[r]))


list_feature_imp(ndf.columns,rfi)


# From the results of randomforest, the least three important features are ```loan_advances```,```director_fees```,```from_messages```.

# ## Use LassoCV to select the least important features

# In[278]:

from sklearn.linear_model import LassoCV


# In[279]:

lcv=LassoCV(max_iter=10000)
lcv.fit(train_X,train_y)


# Lasso does note converge!

# ## Use selectKbest algorithm

# In[280]:

from sklearn.feature_selection import SelectKBest
skb=SelectKBest(k=10)
skb.fit(train_X,train_y)
skb.scores_


# In[281]:

list_feature_imp(ndf.columns,skb.scores_)


# ## Use PCA to select features 

# In[282]:

from sklearn.decomposition import PCA


# In[283]:

pca=PCA(n_components=10,whiten=True)
pca_train_X=pca.fit_transform(train_X,train_y)


# In[284]:

pca.explained_variance_ratio_


# ## Try another fit

# In[285]:

sort_idx=np.argsort(skb.scores_)[::-1]
chosen_idx=sort_idx[0:15]
#nndfmtx=np.hstack((ndfmtx[:,chosen_idx],pca_train_X))
nndfmtx=pca_train_X
#nndfmtx=ndfmtx[:,chosen_idx]
ndf.columns[chosen_idx]


# In[286]:

def print_result(test_y,pred_y):
    result=classification_report(test_y,pred_y)
    print(result)
    print("precision: %s"%precision_score(test_y,pred_y))
    print("recall: %s"%recall_score(test_y,pred_y))


# In[290]:

def try_rf_clf(train_X,train_y,test_X,test_y):
    rf=RandomForestClassifier()
    rf.fit(train_X,train_y)
    pred_y=rf.predict(test_X)
    #print_result(test_y,pred_y)
    return pred_y

def try_nb_clf(train_X,train_y,test_X,test_y):
    lcv=GaussianNB()
    lcv.fit(train_X,train_y)
    pred_y=lcv.predict(test_X)
    #print_result(test_y,pred_y)
    return pred_y
    
def try_lvc_clf(train_X,train_y,test_X,test_y):

    train_X=scale(train_X)

    lvc=LinearSVC(C=0.1)
    lvc.fit(train_X,train_y)
    
    dec_y=lvc.decision_function(train_X)
    
    #choose the smallest 90%
    num_sel=int(len(dec_y)*0.8)
    assert len(dec_y)==train_X.shape[0]
    assert num_sel<=train_X.shape[0]
    
    s_idx=np.argsort(np.abs(dec_y))
    
    assert len(s_idx)==train_X.shape[0]

    for i in s_idx:
        if np.isnan(train_y[i])==True:
            print("smoking index:%s"%i)

    n_train_X=train_X[s_idx[0:num_sel],:]
    n_train_y=train_y[s_idx[0:num_sel]]


    n_train_X=scale(n_train_X)

    lvc.fit(n_train_X,n_train_y)
    

    test_X=scale(test_X)
    pred_y=lvc.predict(test_X)
    return pred_y


# Do the validation

# In[294]:

sss=StratifiedShuffleSplit(label,n_iter=1000,test_size=0.1)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for train_idx,test_idx in sss:

    train_X=ndfmtx[train_idx,:]
    test_X=ndfmtx[test_idx,:]
    train_y=label[train_idx]
    test_y=label[test_idx]
    pred_y=try_lvc_clf(train_X,train_y,test_X,test_y)
    
    for prediction, truth in zip(pred_y, test_y):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        elif prediction == 1 and truth == 1:
            true_positives += 1
        else:
            print("Warning: Found a predicted label not == 0 or 1.")
            print("All predictions should take value 0 or 1.")
            print("Evaluating performance for processed predictions:")
            break

total_predictions = true_negatives + false_negatives + false_positives + true_positives
accuracy = 1.0*(true_positives + true_negatives)/total_predictions
precision = 1.0*true_positives/(true_positives+false_positives)
recall = 1.0*true_positives/(true_positives+false_negatives)
f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)

print("precision: %s"%precision)
print("recall: %s"%recall)
print("f1: %s"%f1)


# In[ ]:



