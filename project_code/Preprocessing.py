# -*- coding: utf-8 -*-
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns


def plots (data):
    fig, axarr1 = plt.subplots(9, 2, figsize=(40, 40))
    #knows the wanted features
    sns.countplot(x ='PhoneService', hue='Churn',data=data , ax=axarr1[0][0])
    sns.countplot(x ='MultipleLines', hue='Churn',data=data , ax=axarr1[0][1])
    sns.countplot(x ='InternetService', hue='Churn',data=data , ax=axarr1[1][0])
    sns.countplot(x ='OnlineSecurity', hue='Churn',data=data , ax=axarr1[1][1])
    sns.countplot(x ='OnlineBackup', hue='Churn',data=data, ax=axarr1[2][0])
    sns.countplot(x ='DeviceProtection', hue='Churn',data=data ,  ax=axarr1[2][1])
    sns.countplot(x ='TechSupport', hue='Churn',data=data , ax=axarr1[3][0])
    sns.countplot(x ='StreamingTV', hue='Churn',data=data, ax=axarr1[3][1])
    sns.countplot(x ='StreamingMovies', hue='Churn',data=data , ax=axarr1[4][0])
    sns.countplot(x ='Contract', hue='Churn',data=data, ax=axarr1[4][1])
    sns.countplot(x ='PaperlessBilling', hue='Churn',data=data, ax=axarr1[5][0])
    sns.countplot(x ='PaymentMethod', hue='Churn',data=data, ax=axarr1[5][1])
    sns.countplot(x ='MonthlyCharges', hue='Churn',data=data, ax=axarr1[6][0])
    sns.countplot(x ='TotalCharges', hue='Churn',data=data, ax=axarr1[6][1])
    sns.countplot(x ='gender', hue='Churn',data=data , ax=axarr1[7][0])
    sns.countplot(x ='SeniorCitizen', hue='Churn',data=data , ax=axarr1[7][1])
    sns.countplot(x ='Partner', hue='Churn',data=data, ax=axarr1[8][0])
    sns.countplot(x ='Dependents', hue='Churn',data=data, ax=axarr1[8][1])


