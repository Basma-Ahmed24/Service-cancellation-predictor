# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox
import numpy as np
from LogisticRegression import *
from Preprocessing import *
from Data_cleaning import *
from SVM import *
from DecisionTree import *
from sklearn.model_selection import train_test_split
import pandas as pd


project=pd.read_csv("CustomersDataset.csv")
# check if there exist any nulls in data
#print(project.isna().sum())
#drop customer id
data = project.drop('customerID', axis=1)
plots(data = data)
data = cleaning(data)
x = data.drop(columns =  ['Churn'])
y = data[['Churn']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=10)



def TakeData():
    data=np.array([senior.get() ,part.get() ,depend.get() , tenur.get() , internet.get() ,
                 onlineS.get() , onlineB.get() , device.get() , tech.get() , streamT.get() 
                   , streamM.get() , contract.get() , paper.get() , payment.get() ,
                   month.get() , total.get()])
    return data

def TrainData():
    if check_value1.get() == 1 :
         global LR
         LR = trainRegression(x_train, y_train)
    if check_value2.get() == 1 :
         global SV
         SV = trainSvm(x_train, y_train)
    if check_value3.get() == 1 :
         global DST
         DST = trainDST(x_train, y_train)


def TestData():
    if check_value1.get() == 1 :
         global LR1
         LR1 = testRegression( LR , x_test, y_test)
    if check_value2.get() == 1 :
         global SV1
         SV1 = testSvm(SV , x_test, y_test)
    if check_value3.get() == 1 :
         global DST1
         DST1 = testDST(DST , x_test, y_test)

def predict_states():
   print("Predict Algorithms:")
   d=TakeData()
   if check_value1.get()==1:
       predictRegression(LR1 , d)
   if check_value2.get()==1:
       predictSvm(SV1 , d)   
   if check_value3.get()==1:
       predictDST(DST1 ,d)  

window = tk.Tk()
window.title("Service Cancellation Predictor")
window.geometry("990x600")

labl0 = tk.Label( text= "Mehtodology" ).grid(row = 0 , column = 3)

check_value1 = tk.IntVar()
check_value2 = tk.IntVar()
check_value3 = tk.IntVar()
check_value1.set(0)
check_value2.set(0)
check_value3.set(0)

check_box1 = tk.Checkbutton( text = "Logistic Regression" , variable = check_value1)
check_box1.grid(row = 1 , column = 0)
check_box2 = tk.Checkbutton( text = "SVM" , variable = check_value2)
check_box2.grid(row = 1 , column = 2)
check_box3 = tk.Checkbutton( text = "ID3" , variable = check_value3)
check_box3.grid(row = 1 , column = 4)


btn1 = tk.Button(text="Train" , width = 20 , command = TrainData).grid(row = 2 , column = 0)
btn2 = tk.Button(text="Test", width = 20  , command = TestData).grid(row = 2 , column = 1)


labl1 = tk.Label( text= "Customer Data" ).grid(row = 4 , column = 0)
labl2 = tk.Label( text= "Customer ID" ).grid(row = 6 , column = 0)
labl3 = tk.Label( text= "Partner" ).grid(row = 8 , column = 0)
labl4 = tk.Label( text= "Phone service" ).grid(row = 10 , column = 0)
labl5 = tk.Label( text= "OnlineSecurity" ).grid(row = 12 , column = 0)
labl6 = tk.Label( text= "TechSupport" ).grid(row = 14 , column = 0)
labl7 = tk.Label( text= "Contract" ).grid(row = 16 , column = 0)
labl8 = tk.Label( text= "MonthlyCharges" ).grid(row = 18 , column = 0)

cus_ID = tk.Entry(window,bd = 2 )
cus_ID.grid(row = 6 , column = 1)
part = tk.Entry(bd = 2)
part.grid(row = 8 , column = 1)
Phone = tk.Entry(bd = 2)
Phone.grid(row = 10 , column = 1)
onlineS = tk.Entry(bd = 2)
onlineS.grid(row = 12 , column = 1)
tech = tk.Entry(bd = 2 )
tech.grid(row = 14 , column = 1)
contract = tk.Entry(bd = 2 )
contract.grid(row = 16 , column = 1)
month = tk.Entry(bd = 2 )
month.grid(row = 18 , column = 1)


labl9 = tk.Label( text= "gender" ).grid(row = 6 , column = 3)
labl10 = tk.Label( text= "Dependents" ).grid(row = 8 , column = 3)
labl11 = tk.Label( text= "MultipleLines" ).grid(row = 10 , column = 3)
labl12 = tk.Label( text= "OnlineBackup" ).grid(row = 12 , column = 3)
labl13 = tk.Label( text= "StreamingTV" ).grid(row = 14 , column = 3)
labl14 = tk.Label( text= "PaperlessBilling" ).grid(row = 16 , column = 3)
labl15 = tk.Label( text= "TotalCharges" ).grid(row = 18 , column = 3)

gender = tk.Entry(bd = 2)
gender.grid(row = 6 , column = 4)
depend = tk.Entry(bd = 2)
depend.grid(row = 8 , column = 4)
multi = tk.Entry(bd = 2)
multi.grid(row = 10 , column = 4)
onlineB = tk.Entry(bd = 2 )
onlineB.grid(row = 12 , column = 4)
streamT = tk.Entry(bd = 2)
streamT.grid(row = 14 , column = 4)
paper = tk.Entry(bd = 2)
paper.grid(row = 16 , column = 4)
total = tk.Entry(bd = 2 )
total.grid(row = 18 , column = 4)


labl16 = tk.Label( text= "SeniorCitizen" ).grid(row = 6 , column = 7)
labl17 = tk.Label( text= "tenure" ).grid(row = 8 , column = 7)
labl18 = tk.Label( text= "InternetService" ).grid(row = 10 , column = 7)
labl19 = tk.Label( text= "DeviceProtection" ).grid(row = 12 , column = 7)
labl20 = tk.Label( text= "StreamingMovies" ).grid(row = 14 , column = 7)
labl21 = tk.Label( text= "PaymentMethod" ).grid(row = 16 , column = 7)

senior = tk.Entry(bd = 2 )
senior.grid(row = 6 , column = 8)
tenur = tk.Entry(bd = 2 )
tenur.grid(row = 8 , column = 8)
internet = tk.Entry(bd = 2 )
internet.grid(row = 10 , column = 8)
device = tk.Entry(bd = 2)
device.grid(row = 12 , column = 8)
streamM = tk.Entry(bd = 2)
streamM.grid(row = 14 , column = 8)
payment = tk.Entry(bd = 2 )
payment.grid(row = 16 , column = 8)


btn2 = tk.Button(text="Pridect",command=predict_states, width = 20).grid(row = 22 , column = 3)

window.mainloop()