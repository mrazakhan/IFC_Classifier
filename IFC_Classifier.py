from __future__ import division
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,roc_curve,auc
from sklearn import metrics
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cross_validation import KFold,train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import os

nFolds=2


class Logger(object):
    def __init__(self,log_file="log.dat"):
        self.terminal = sys.stdout
        self.log = open(log_file, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

def get_test_pred(clf,X_test):
    # This function returns prediction and probabilities for test data
    return clf.predict(X_test),clf.predict_proba(X_test)


def run_cv(out_dir,X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=nFolds,shuffle=True)
    y_pred = y.copy()
    i=0
    y_prob = np.zeros((len(y),2))
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
        y_prob[test_index]=clf.predict_proba(X_test)
        i=i+1
    return clf,y_pred,y_prob


def accuracy(y_true,y_pred):
    # NumPy interpretes True and False as 1. and 0.
    return np.mean(y_true == y_pred)


def draw_confusion_matrices(out_dir,confusion_matrices, class_names):
    labels = list(class_names)
    for cm in confusion_matrices:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm[1])
        pl.title('Confusion Matrix (%s)' % cm[0])
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        pl.xlabel('Predicted Class')
        pl.ylabel('True Class')
        for i,j in ((x,y) for x in xrange(len(cm[1])) for y in xrange(len(cm[1][0]))):
            ax.annotate(str(cm[1][i][j]), xy=(i,j), color='white')
        pl.savefig(out_dir+"/"+cm[0])
        
 
def draw_ROC_curves(ROC_Curves,filename):
    fig=plt.figure()
    plt.clf()
    from itertools import cycle
    markers = [".","+","x","^","v","*"]
	
    markercycler=cycle(markers)

    liness=['_', '-', '--', ':']
    linestylecycler=cycle(liness)
    for rc in ROC_Curves:
        fpr, tpr, thresholds = rc[1]
        # Compute the area under the ROC curve
        roc_auc = auc(fpr, tpr)
        print("Area under the ROC curve : %f" % roc_auc)
        # Plot the ROC curv
        plt.plot(fpr, tpr,linewidth=1, linestyle=next(linestylecycler),marker=next(markercycler),label=' %s (area = %0.2f)' %(rc[0], roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
	outstring = zip(fpr, tpr)
	f = open(out_dir+"/"+rc[0]+'ROCData_FPR_TPR.txt', 'w')
	for line in outstring:
    		f.write(" ".join(str(x) for x in line) + "\n")
	f.close()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('')
    plt.legend(loc="lower right")
    fig.savefig(out_dir+"/"+filename)

def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
             df.apply(np.random.shuffle, axis=axis)
    return df
    
if __name__ == "__main__":

    if len(sys.argv)!=6:
            print "Number of arguments passed", len(sys.argv)
            print "Required Input arguments : Input_DataFrame Log_File Out_Dir Features_File MMUserType(0 for Inactive, 1 forActive)"
            sys.exit(-1)


    input_df=sys.argv[1]
    log_file=sys.argv[2]
    out_dir=sys.argv[3]
    features_file=sys.argv[4]
    i=int(sys.argv[5])
    
    #i=1
    #input_df=input_data[i]
    #log_file=log_files[i]
    #out_dir=out_dirs[i]

    #features_file=features_files[i]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    iFile=input_df
    print "Reading input file", iFile
    sys.stdout = Logger(out_dir+"/"+log_file)
    churn_df=pd.read_csv(iFile)
   
    #churn_df=shuffle(churn_df)

    col_names=churn_df.columns.tolist()
    print "Converting Y to 1 in the labels"
    churn_result = churn_df['UserType']
    try:
    	y = np.where(churn_result == 'normal',0,1)
    except:
	y=churn_result
	pass
    print "Dropping unwanted cols"
    to_drop=['CallerId','UserType']
    #to_drop=['X','X.1','CallerId','UserType','sq-X.1','sqrt-X.1','log2-X.1']
    to_keep=[]
    MMUsersType=["MM","ActiveMM"]

    with open(features_file) as feat_file:
        j=0
        for line in feat_file:
            j=j+1
            try:
                    name=line.split(',')[0]
                    to_keep.append(name.strip())
            except Exception as inst:
                    print ' Exception in line', j
                    print type(inst)     # the exception instance
                    print inst.args      # arguments stored in .args
                    print inst  
                    #sys.exit(-1)
    #print i,name,var

        

    print 'To Keep length',len(to_keep)
    
    print 'To Drop length',len(to_drop)

    churn_feat_labels=col_names

    for feat in churn_feat_labels:
	if feat not in to_keep:
		to_drop.append(feat)
    churn_feat_space = churn_df.drop(to_drop,axis=1)


    print "Converting column types to floats"
    X = churn_feat_space.as_matrix().astype(np.float)
    print "Scaling Data"
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_Train,X_Test,y_train,y_test=train_test_split(X,y)

    print "Feature space holds %d observations and %d features" % X_Train.shape
    print "Unique target labels:", np.unique(y_train)

    print "Applying SVM"
    svmc,SVC_Train_Result,SVC_Train_Prob=run_cv(out_dir,X_Train,y_train,SVC,probability=True)

    SVC_Test_Result,SVC_Test_Prob=get_test_pred(svmc,X_Test)


    print "Applying RF"
    rfc,RF_Train_Result,RF_Train_Prob=run_cv(out_dir,X_Train,y_train,RF)

    RF_Test_Result,RF_Test_Prob=get_test_pred(rfc,X_Test)

    print "Applying KNN"
    knnc,KNN_Train_Result,KNN_Train_Prob=run_cv(out_dir,X_Train,y_train,KNN)

    KNN_Test_Result,KNN_Test_Prob=get_test_pred(knnc,X_Test)

    print "Applying AdaBoost"
    adac,AdaBoost_Train_Result,AdaBoost_Train_Prob=run_cv(out_dir,X_Train,y_train,AdaBoostClassifier,n_estimators=200)

    AdaBoost_Test_Result,AdaBoost_Test_Prob=get_test_pred(adac,X_Test)

    print "Applying GradBoost"
    gdmc,GradBoost_Train_Result,GradBoost_Train_Prob=run_cv(out_dir,X_Train,y_train,GradientBoostingClassifier,n_estimators=200)

    GradBoost_Test_Result,GradBoost_Test_Prob=get_test_pred(gdmc,X_Test)

    print "Applying Log Regression"
    logc,Log_Train_Result,Log_Train_Prob=run_cv(out_dir,X_Train,y_train,LogisticRegression)

    Log_Test_Result,Log_Test_Prob=get_test_pred(logc,X_Test)

    #print "Support vector machines:"
    print "SVM Train %.4f" % accuracy(y_train, SVC_Train_Result)
    output_df=pd.DataFrame(y_train,columns=["Original"])
    output_df['Predicted']=SVC_Train_Result
    output_df.to_csv(out_dir+"/"+"SVC_TrainReport.csv")
    SVC_Train_Prob.tofile(out_dir+'/'+'SVC_Prob.txt',sep=';'  )

    print "SVM Test %.4f" % accuracy(y_test, SVC_Test_Result)
    output_df=pd.DataFrame(y_test,columns=["Original"])
    output_df['Predicted']=SVC_Test_Result
    output_df.to_csv(out_dir+"/"+"SVC_TestReport.csv")
    SVC_Test_Prob.tofile(out_dir+'/'+'SVC_Test_Prob.txt',sep=';'  )


    print "RF Train %.4f" % accuracy(y_train, RF_Train_Result)
    output_df=pd.DataFrame(y_train,columns=["Original"])
    output_df['Predicted']=RF_Train_Result
    output_df.to_csv(out_dir+"/"+"RF_TrainReport.csv")
    RF_Train_Prob.tofile(out_dir+'/'+'RF_Prob.txt',sep=';'  )

    print "RF Test %.4f" % accuracy(y_test, RF_Test_Result)
    output_df=pd.DataFrame(y_test,columns=["Original"])
    output_df['Predicted']=RF_Test_Result
    output_df.to_csv(out_dir+"/"+"RF_TestReport.csv")
    RF_Test_Prob.tofile(out_dir+'/'+'RF_Test_Prob.txt',sep=';'  )


    print "KNN Train %.4f" % accuracy(y_train, KNN_Train_Result)
    output_df=pd.DataFrame(y_train,columns=["Original"])
    output_df['Predicted']=KNN_Train_Result
    output_df.to_csv(out_dir+"/"+"KNN_TrainReport.csv")
    KNN_Train_Prob.tofile(out_dir+'/'+'KNN_Prob.txt',sep=';'  )

    print "KNN Test %.4f" % accuracy(y_test, KNN_Test_Result)
    output_df=pd.DataFrame(y_test,columns=["Original"])
    output_df['Predicted']=KNN_Test_Result
    output_df.to_csv(out_dir+"/"+"KNN_TestReport.csv")
    KNN_Test_Prob.tofile(out_dir+'/'+'KNN_Test_Prob.txt',sep=';'  )

    print "AdaBoost Train %.4f" % accuracy(y_train, AdaBoost_Train_Result)
    output_df=pd.DataFrame(y_train,columns=["Original"])
    output_df['Predicted']=AdaBoost_Train_Result
    output_df.to_csv(out_dir+"/"+"AdaBoost_TrainReport.csv")
    AdaBoost_Train_Prob.tofile(out_dir+'/'+'AdaBoost_Prob.txt',sep=';'  )


    print "AdaBoost Test %.4f" % accuracy(y_test, AdaBoost_Test_Result)
    output_df=pd.DataFrame(y_test,columns=["Original"])
    output_df['Predicted']=AdaBoost_Test_Result
    output_df.to_csv(out_dir+"/"+"AdaBoost_TestReport.csv")
    AdaBoost_Test_Prob.tofile(out_dir+'/'+'AdaBoost_Test_Prob.txt',sep=';'  )

    print "Log Train %.4f" % accuracy(y_train, Log_Train_Result)
    output_df=pd.DataFrame(y_train,columns=["Original"])
    output_df['Predicted']=Log_Train_Result
    output_df.to_csv(out_dir+"/"+"Log_TrainReport.csv")
    Log_Train_Prob.tofile(out_dir+'/'+'Log_Prob.txt',sep=';'  )

    print "Log Test %.4f" % accuracy(y_test, Log_Test_Result)
    output_df=pd.DataFrame(y_test,columns=["Original"])
    output_df['Predicted']=Log_Test_Result
    output_df.to_csv(out_dir+"/"+"Log_TestReport.csv")
    Log_Test_Prob.tofile(out_dir+'/'+'Log_Test_Prob.txt',sep=';'  )

    print "GradBoost Train %.4f" % accuracy(y_train, GradBoost_Train_Result)
    output_df=pd.DataFrame(y_train,columns=["Original"])
    output_df['Predicted']=GradBoost_Train_Result
    output_df.to_csv(out_dir+"/"+"GradBoost_TrainReport.csv")
    GradBoost_Train_Prob.tofile(out_dir+'/'+'GradBoost_Prob.txt',sep=';'  )


    print "GradBoost Test %.4f" % accuracy(y_test, GradBoost_Test_Result)
    output_df=pd.DataFrame(y_test,columns=["Original"])
    output_df['Predicted']=GradBoost_Test_Result
    output_df.to_csv(out_dir+"/"+"GradBoost_TestReport.csv")
    GradBoost_Test_Prob.tofile(out_dir+'/'+'GradBoost_Test_Prob.txt',sep=';'  )


    y_train = np.array(y_train)
    class_names = np.unique(y_train)

    confusion_matrices_training = [
            ( "Support Vector Machines Training", confusion_matrix(y_train,SVC_Train_Result) ),
            ( "Random Forest Training", confusion_matrix(y_train,RF_Train_Result) ),
            ( "K-Nearest-Neighbors Training", confusion_matrix(y_train,KNN_Train_Result) ),
            ("AdaBoost-Result Training",confusion_matrix(y_train,AdaBoost_Train_Result)),
            ("Logistic-Result Training",confusion_matrix(y_train,Log_Train_Result)),
            ("GradBoost-Result Training",confusion_matrix(y_train,GradBoost_Train_Result))
    ]

    draw_confusion_matrices(out_dir,confusion_matrices_training,class_names)

    y_test = np.array(y_test)
    class_names = np.unique(y_test)


    confusion_matrices_testing = [
            ( "Support Vector Machines Testing", confusion_matrix(y_test,SVC_Test_Result) ),
            ( "Random Forest Testing", confusion_matrix(y_test,RF_Test_Result) ),
            ( "K-Nearest-Neighbors Testing", confusion_matrix(y_test,KNN_Test_Result) ),
            ("AdaBoost-Result Testing",confusion_matrix(y_test,AdaBoost_Test_Result)),
            ("Logistic-Result Testing",confusion_matrix(y_test,Log_Test_Result)),
            ("GradBoost-Result Testing",confusion_matrix(y_test,GradBoost_Test_Result))
    ]


    draw_confusion_matrices(out_dir,confusion_matrices_testing,class_names)

    print 'SVM Classification Training Report'
    print metrics.classification_report(y_train, SVC_Train_Result, target_names=['Normal', MMUsersType[i]])

    print 'RF Classification Training Report'
    print metrics.classification_report(y_train, RF_Train_Result, target_names=['Normal', MMUsersType[i]])

    print 'KNN Classification Training Report'
    print metrics.classification_report(y_train, KNN_Train_Result, target_names=['Normal', MMUsersType[i]])

    print 'AdaBoost Training Report'
    print metrics.classification_report(y_train, AdaBoost_Train_Result, target_names=['Normal', MMUsersType[i]])

    print 'Log Reg Training Report'
    print metrics.classification_report(y_train, Log_Train_Result, target_names=['Normal', MMUsersType[i]])

    print 'GradBoost Training Report'
    print metrics.classification_report(y_train, GradBoost_Train_Result, target_names=['Normal', MMUsersType[i]])


    print 'SVM Classification Testing Report'
    print metrics.classification_report(y_test, SVC_Test_Result, target_names=['Normal', MMUsersType[i]])

    print 'RF Classification Testing Report'
    print metrics.classification_report(y_test, RF_Test_Result, target_names=['Normal', MMUsersType[i]])

    print 'KNN Classification Testing Report'
    print metrics.classification_report(y_test, KNN_Test_Result, target_names=['Normal', MMUsersType[i]])

    print 'AdaBoost Testing Report'
    print metrics.classification_report(y_test, AdaBoost_Test_Result, target_names=['Normal', MMUsersType[i]])

    print 'Log Reg Testing Report'
    print metrics.classification_report(y_test, Log_Test_Result, target_names=['Normal', MMUsersType[i]])

    print 'GradBoost Testing Report'
    print metrics.classification_report(y_test, GradBoost_Test_Result, target_names=['Normal', MMUsersType[i]])


    ROC_Curves_Training=[('SVM',roc_curve(y_train,SVC_Train_Prob[:,1])),
            #('Random Forest',roc_curve(y_train,RF_Train_Prob[:,1])),
            #('KNN',roc_curve(y_train,KNN_Train_Prob[:,1])),
            #('AdaBoost',roc_curve(y_train,AdaBoost_Train_Prob[:,1])),
            ('Logistic Regression',roc_curve(y_train,Log_Train_Prob[:,1])),
            ('GradBoost',roc_curve(y_train,GradBoost_Train_Prob[:,1]))
            ]
    draw_ROC_curves(ROC_Curves_Training,"ROC_Curves_Training_MM.png")



    ROC_Curves_Testing=[('SVM',roc_curve(y_test,SVC_Test_Prob[:,1])),
            #('Random Forest',roc_curve(y_test,RF_Test_Prob[:,1])),
            #('KNN',roc_curve(y_test,KNN_Test_Prob[:,1])),
            #('AdaBoost',roc_curve(y_test,AdaBoost_Test_Prob[:,1])),
            ('Logistic Regression',roc_curve(y_test,Log_Test_Prob[:,1])),
            ('GradBoost',roc_curve(y_test,GradBoost_Test_Prob[:,1]))
            ]
    draw_ROC_curves(ROC_Curves_Testing,"ROC_Curves_Testing_MM.png")

