import numpy as np
from sklearn import preprocessing as pp



#change datatype of columns and convert the categorical to numeric
def cleaning (data):
    
    label_encoder=pp.LabelEncoder()
    data['Partner']= label_encoder.fit_transform(data['Partner'])
   
    data["gender"]= label_encoder.fit_transform(data['gender'])
    
    data["Dependents"]= label_encoder.fit_transform(data['Dependents'])
    data["InternetService"]= label_encoder.fit_transform(data['InternetService'])
    data["OnlineSecurity"]= label_encoder.fit_transform(data['OnlineSecurity'])
    data["Churn"]= label_encoder.fit_transform(data['Churn'])
    data["MultipleLines"]= label_encoder.fit_transform(data['MultipleLines'])
    data["OnlineSecurity"]= label_encoder.fit_transform(data['OnlineSecurity'])
    data["OnlineBackup"]= label_encoder.fit_transform(data['OnlineBackup'])
    data["DeviceProtection"]= label_encoder.fit_transform(data['DeviceProtection'])
    data["TechSupport"]= label_encoder.fit_transform(data['TechSupport'])
    data["StreamingTV"]= label_encoder.fit_transform(data['StreamingTV'])
    data["StreamingMovies"]= label_encoder.fit_transform(data['StreamingMovies'])
    data["InternetService"]= label_encoder.fit_transform(data['InternetService'])   
    data["Contract"]= label_encoder.fit_transform(data['Contract']) 
    data["PaymentMethod"]= label_encoder.fit_transform(data['PaymentMethod']) 
    data["PaperlessBilling"]= label_encoder.fit_transform(data['PaperlessBilling']) 

#convert the empty cells to nan , changing data type and fill all nan values by using the mean of the column

    data["TotalCharges"] = data["TotalCharges"].replace(" " , np.nan)
    data["TotalCharges"]=data["TotalCharges"].astype('float64')
    data["TotalCharges"]=data["TotalCharges"].fillna(value= data["TotalCharges"].mean())

#normalization of data

    data_scaler= pp.MinMaxScaler(feature_range=(0 , 1))
    TotalCharges_array=data[["TotalCharges"]]
    TotalCharges = data_scaler.fit_transform(TotalCharges_array)
    data["TotalCharges"] = TotalCharges
    
    MonthlyCharges_array=data[["MonthlyCharges"]]
    MonthlyCharges = data_scaler.fit_transform(MonthlyCharges_array)
    data["MonthlyCharges"] = MonthlyCharges
    
    tenure_array=data[["tenure"]]
    tenure = data_scaler.fit_transform(tenure_array)
    data["tenure"] = tenure
    
#drop the unwanted features
    
    data = data.drop('gender', axis=1)
    data = data.drop('PhoneService', axis=1)
    data = data.drop('MultipleLines', axis=1)
  
    #print ('inforamtion: ')
    #print (data.info())
    #print ('description: ')
    #print (data.describe())
    
    return data



