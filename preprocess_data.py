__author__ = 'kanhua'


import pickle
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin,TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import scale
import copy

def pkl_to_df(pkl_file="./data/final_project_dataset.pkl", rows_to_remove=["TOTAL"], cols_to_remove=["email_address"]):

    data_dict=pickle.load(open(pkl_file,"rb"),fix_imports=False,encoding="latin1")

    df=pd.DataFrame(data_dict)
    df=df.transpose()

    df=df.drop(rows_to_remove,axis=0)
    df=df.drop(cols_to_remove,axis=1)

    return df

def extract_df(df,exclude_features=["poi"],verbose=False):
    """
    :param df: input dataframe
    :param exclude_features: list of feature names to be removed
    :return: a tuple of X,y
    """

    y=df["poi"].values

    df=df.drop(["poi"],axis=1)

    X=df.values



    if verbose==True:
        print(df.columns)

    return X,y,df.columns

def linearsvc_outlier_rm(train_X,train_y,discard=0.1,lvc_C=0.1,take_abs=True):
    """
    Remove the outliers in the data.
    It rescaled the data, use linear SVC to do the classification,
    and then remove the data with farthest distances
    :param train_X: train data
    :param train_y: label
    :param discard: the ratio of the outliers to be removed
    :return: tuple of new X,y
    """

    assert isinstance(train_X,np.ndarray)
    assert isinstance(train_y,np.ndarray)

    # LinearSVC requires the features to be scaled
    # Here we scaled the input data, but the output data are note rescaled
    scaled_train_X=scale(train_X)

    lvc=LinearSVC(C=lvc_C)
    lvc.fit(scaled_train_X,train_y)

    dec_y=lvc.decision_function(scaled_train_X)

    #choose the smallest 90%
    num_sel=int(len(dec_y)*(1-discard))
    assert len(dec_y)==scaled_train_X.shape[0]
    assert num_sel<=scaled_train_X.shape[0]

    if take_abs==True:
        s_idx=np.argsort(np.abs(dec_y))
    else:
        s_idx=np.argsort(dec_y)

    assert len(s_idx)==scaled_train_X.shape[0]


    n_train_X=train_X[s_idx[0:num_sel],:]
    n_train_y=train_y[s_idx[0:num_sel]]

    return n_train_X,n_train_y,dec_y


class FeatureSel(BaseEstimator,TransformerMixin):
    def __init__(self,k_best=5,pca_comp=8):
        self.k_best=k_best
        self.pca_comp=pca_comp
        if pca_comp>0:
            self.pca=PCA(n_components=self.pca_comp)
        if k_best>0:
            self.skb=SelectKBest(k=self.k_best)


    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

        self.pca.set_params(n_components=self.pca_comp)

        self.skb.set_params(k=self.k_best)

        return self


    def transform(self,X):
        X1=self.pca.transform(X)
        X2=self.skb.transform(X)

        return np.hstack((X1,X2))


    def fit_transform(self,X,y):


        X1=self.pca.fit_transform(X,y)
        X2=self.skb.fit_transform(X,y)

        return np.hstack((X1,X2))

    def fit(self,X,y):
        if self.pca_comp>0:
            self.pca.fit(X,y)
        if self.k_best>0:
            self.skb.fit(X,y)


def add_features(data_dict, financial_features="none"):
    '''
    This function takes separate positive and negative values of financial features,
    and then it takes logarithm of the values of each feature.
    :param financial_features: will be set to default value if it is "none"
    :param data_dict: the data dictionary
    :return: data dictionary with new features, names of new features, names of financial features
    '''

    if financial_features=="none":
        financial_features = ['bonus', 'deferral_payments', 'deferred_income', 'director_fees',
                              'exercised_stock_options', 'expenses',
                              'long_term_incentive', 'other', 'restricted_stock',
                              'restricted_stock_deferred', 'salary',
                              'total_payments', 'total_stock_value']

    new_data_dict = copy.copy(data_dict)

    for name in new_data_dict.keys():
        for f in financial_features:
            if new_data_dict[name][f] == "NaN":
                new_data_dict[name]["p_" + f] = 0
                new_data_dict[name]["n_" + f] = 0
            elif new_data_dict[name][f] >= 0:
                new_data_dict[name]["p_" + f] = np.log10(new_data_dict[name][f])
                new_data_dict[name]["n_" + f] = 0
            elif new_data_dict[name][f] < 0:
                new_data_dict[name]["n_" + f] = np.log10(-new_data_dict[name][f])
                new_data_dict[name]["p_" + f] = 0

    new_feature = []

    for f in financial_features:
        new_feature.append("n_" + f)
        new_feature.append("p_" + f)

    return new_data_dict, new_feature, financial_features




if __name__=="__main__":
    df= pkl_to_df()
    X,y,_=extract_df(df)



    from sklearn.preprocessing import Imputer
    import matplotlib.pyplot as plt

    imp=Imputer(axis=0,strategy="median")
    X=imp.fit_transform(X)

    X,y,y_dis=linearsvc_outlier_rm(X,y)
    print(y_dis)
    plt.hist(y_dis,100,hold=True)


    X,y,y_dis=linearsvc_outlier_rm(X,y)
    print(y_dis)
    plt.hist(y_dis,100)


    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    sd=StandardScaler()
    fsl=FeatureSel()
    fsl.set_params(k_best=1)
    ppl=Pipeline([("fsl",fsl),("sd",sd),("lvc",LinearSVC(C=0.0001,tol=0.0001))])

    ppl.fit(X,y)

    ppl.predict(X)







