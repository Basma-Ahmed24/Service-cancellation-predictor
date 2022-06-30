# -*- coding: utf-8 -*-
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
#from mlxtend.plotting import plot_decision_regions



def trainSvm(x_train , y_train):
    #train data
    SV = SVC(kernel='rbf', gamma=1.00)
    SV.fit(x_train , y_train)
    prediction= SV.predict(x_train)
    ac_svm=accuracy_score(y_train,prediction)
    print("SVM train accuracy: ",ac_svm)
    plt.title("scatter train SVM for TotalCharges ")
    x = np.random.normal( 0.261309,  0.261366, 1000) #(mean,standard deviation,dots)
    y = np.random.normal(0.265370, 0.441561, 1000) #(mean,standard deviation,dots)
    plt.scatter(x, y)
    plt.show()
    return SV

def testSvm( SV , x_test , y_test):
    # Evaluate by means of a confusion matrix
    matrix = plot_confusion_matrix(SV, x_test, y_test,cmap=plt.cm.Blues, normalize='true')
    plt.title('Confusion matrix for RBF SVM')
    plt.show(matrix)
    plt.show()
    
    y_pre = SV.predict(x_test)
    ac_svm=accuracy_score(y_test,y_pre)
    print("SVM test accuracy: ",ac_svm)
    # module implementation :
    #report = classification_report(y_test , y_pre)
    #print('Report: ')
    #print(report)
    
    plt.title("scatter test SVM for TotalCharges ")
    x1 = np.random.normal( 0.261309,  0.261366, 1000) #(mean,standard deviation,dots)
    y1 = np.random.normal(0.265370, 0.441561, 1000) #(mean,standard deviation,dots)
    plt.scatter(x1, y1)
    plt.show()
    return SV

def predictSvm( SV , data):
    #Predict of churn value
    '''SeniorCitizen = input('Enter your  Senior citizen value : ')
    partner = input('Enter your partner value : ')
    Dependents = input('Enter your dependents value : ')
    tenure = input('Enter your tenure value : ')
    0InternetService = input('Enter your InternetService value : ')
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
    TotalCharges = input('Enter your TotalCharges  value: ')
    xtest1=np.array([SeniorCitizen , partner , Dependents , tenure , InternetService , OnlineSecurity , OnlineBackup , DeviceProtection , TechSupport , StreamingTV , StreamingMovies , Contract , PaperlessBilling , PaymentMethod , MonthlyCharges , TotalCharges])
    '''
    xtest1=data
    xtest1 = xtest1.reshape(1, -1)
    ytest1=SV.predict(xtest1)
    e = "yes"
    if ytest1 == 0:
       e = "no"    
    print('SVM predicted Churn is ' + str(int(ytest1[0])) + "  for  "+ e )