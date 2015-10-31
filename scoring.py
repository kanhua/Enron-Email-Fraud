__author__ = 'kanhua'


from sklearn.cross_validation import StratifiedShuffleSplit


def calc_score(X,y,clf,n_iter=1000):

    sss=StratifiedShuffleSplit(y,n_iter=1000)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx,test_idx in sss:

        train_X=X[train_idx,:]
        test_X=X[test_idx,:]
        train_y=y[train_idx]
        test_y=y[test_idx]
        clf.fit(train_X,train_y)
        pred_y=clf.predict(test_X)

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

    return {"precision":precision,"recall":recall,"f1":f1}


def test_feature_selection(test_df):
    fill_test_df=test_df.fillna(0)
    X,y,cols=extract_df(fill_test_df)
    skb=SelectKBest()
    nX=skb.fit_transform(X,y)
    skb.scores_
    sorted_idx=np.argsort(skb.scores_)[::-1]
    cols[sorted_idx]
    calc_score(nX,y,quickclf)