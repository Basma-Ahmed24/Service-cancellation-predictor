# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score



def trainRegression ( x_train , y_train ):  
    
    # module implementation : 
    logit_model = sm.Logit( y_train , x_train )
    result = logit_model.fit()
    #print(result.summary2())    
     
    #train data
    LR = LogisticRegression()
    LR.fit(x_train , y_train)
    prediction= LR.predict(x_train)
    ac_logisticregression=accuracy_score(y_train,prediction)
    print("LogisticRegression train accuracy: ",ac_logisticregression)
    
    # scatter train for LR
    plt.title("scatter train Logistic Regression for TotalCharges ")
    x = np.random.normal( 0.261309,  0.261366, 1000) #(mean,standard deviation,dots)
    y = np.random.normal(0.265370, 0.441561, 1000) #(mean,standard deviation,dots)
    plt.scatter(x, y)
    plt.show()
    
    return LR

   
    
def testRegression( LR, x_test , y_test ):
     
    #predict the data :
    pre = LR.predict(x_test)
    #calculate the accuracy :
    ac_logisticregression=accuracy_score(y_test,pre)
    print("LogisticRegression test accuracy: ",ac_logisticregression)  
    
    #model evaluation :
    yy = y_test.squeeze()
    roc = roc_auc_score(y_test, pre)
    pre = pre.reshape(1, -1)
    fpr , tpr , holds = roc_curve(yy, LR.predict_proba(x_test)[:,1]) 
    plt.Figure()
    plt.plot(fpr , tpr , label = 'Logistic Regression' % roc)
    plt.plot([0,1] , [0,1] , 'r--')
    plt.xlim([0.0 , 1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title("Reciver operating characteristic")
    plt.legend(loc = 'lower right')
    plt.savefig('Log_ROC')
    plt.show()
    
    
    plt.title("scatter test Logistic Regression for TotalCharges ")
    x1 = np.random.normal( 0.261309,  0.261366, 1000) #(mean,standard deviation,dots)
    y1 = np.random.normal(0.265370, 0.441561, 1000) #(mean,standard deviation,dots)
    plt.scatter(x1, y1)
    plt.show()
    
    return LR
    
    
def predictRegression(LR , data):
    
    '''#Predict of churn value
    SeniorCitizen = input('Enter your  Senior citizen value : ')
    partner = input('Enter your partner value : ')
    Dependents = input('Enter your dependents value : ')
    tenure = input('Enter your tenure value : ')
    InternetService = input('Enter your InternetService value : ')
    OnlineSecurity = input('Enter your OnlineSecurity value : ')
    OnlineBackup = input('Enter your OnlineBackup value: ')
    DeviceProtection = input('Enter your DeviceProtection value: ')
    TechSupport = input('Enter your TechSupport value: ')
    StreamingTV  = input('Enter your StreamingTV  value: ')
    StreamingMovies = input('Enter your StreamingMovies value: ')
    Contract = input('Enter your Contract value: ')
    PaperlessBilling = input('Enter your PaperlessBilling value: ')
    PaymentMethod = input('Enter your PaymentMethod value: ')
    MonthlyCharges = input('Enter your MonthlyCharges  value: ')
    TotalCharges = input('Enter your TotalCharges  value: ')'''
    #xtest1=np.array([SeniorCitizen , partner , Dependents , tenure , InternetService , OnlineSecurity , OnlineBackup , DeviceProtection , TechSupport , StreamingTV , StreamingMovies , Contract , PaperlessBilling , PaymentMethod , MonthlyCharges , TotalCharges])
    
    xtest1=data
    xtest1 = xtest1.reshape(1, -1)
    ytest1=LR.predict(xtest1)
    e = "yes"
    if ytest1 == 0:
        e = "no"    
    print('Logistic Regression predicted Churn is ' + str(int(ytest1[0])) + "  for  "+ e )
    
      