import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



def trainDST(x_train , y_train):
    model_DecTree = DecisionTreeClassifier(criterion = "gini", random_state = 10,
                                          max_depth=10, min_samples_leaf=12)
    model_DecTree.fit(x_train , y_train)
    prediction= model_DecTree.predict(x_train)
    ac_id3=accuracy_score(y_train,prediction)
    print("Decision Tree train accuracy: ",ac_id3)
    return model_DecTree

def testDST( model_DecTree , x_test , y_test):
    y_predict = model_DecTree.predict(x_test)
    ac=accuracy_score(y_test,y_predict)
    print('Decision Tree test Accuracy : ' , ac)
    return model_DecTree

def predictDST(model_DecTree , data):
    '''SeniorCitizen = input('Enter your  Senior citizen value : ')
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
    TotalCharges = input('Enter your TotalCharges  value: ')
    xtest1=np.array([SeniorCitizen , partner , Dependents , tenure , InternetService , OnlineSecurity , OnlineBackup , DeviceProtection , TechSupport , StreamingTV , StreamingMovies , Contract , PaperlessBilling , PaymentMethod , MonthlyCharges , TotalCharges])
    '''
    xtest1=data
    xtest1 = xtest1.reshape(1, -1)
    ytest1=model_DecTree.predict(xtest1)
    e = "yes"
    if ytest1 == 0:
        e = "no"    
    print('Decision Tree predicted Churn is ' + str(int(ytest1[0])) + "  for  "+ e )