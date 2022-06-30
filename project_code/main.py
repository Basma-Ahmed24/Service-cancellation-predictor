# -*- coding: utf-8 -*-
from LogisticRegression import *
from Preprocessing import *
from Data_cleaning import *
from SVM import *
from DecisionTree import *
from sklearn.model_selection import train_test_split
import pandas as pd


project=pd.read_csv("CustomersDataset.csv")
# check if there exist any nulls in data
#project.isna().sum()
#drop customer id
data = project.drop('customerID', axis=1)

#plots(data = data)

data = cleaning(data)
#print(data)
#split the data into 80% training and 20% testing
    
x = data.drop(columns =  ['Churn'])
y = data[['Churn']]
#x = pp.StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=10)
    
   
# print('do you want to predict with : ')
# print( '1 - logistic regression ')
# print('2 - SVM')
# print('3 - DecisionTree')
# decision = input("enter '1' for logistic or '2' for SVM or '3' for decisiontree: ")

'''
if decision == '1' :
   # logistic regression : 
   # train data :   
   LR = trainRegression(x_train , y_train )
   # test data :
   LR = testRegression(LR , x_test , y_test )
   # prediction new data : 
   predictRegression(LR)

elif decision == '2' :
    SV = trainSvm(x_train , y_train)
    SV = testSvm(SV, x_test, y_test)
    predictSvm(SV)   
elif decision == '3' :
    model_DecTree = trainDST( x_train, y_train)
    model_DecTree = testDST(model_DecTree, x_test, y_test)
    predictDST(model_DecTree) 
'''

# #train data
# print("train regression")
# LR = trainRegression(x_train, y_train)
# print("train svm")
# SV = trainSvm(x_train , y_train)
# print("train tree")
#TR = trainDST(x_train, y_train)    
# #test
# print("test regression")
# LR = testRegression(LR, x_test, y_test)
# print("test svm")
# SV = testSvm(SV , x_test , y_test)
# print("test tree")
#TR = testDST(TR, x_test, y_test)