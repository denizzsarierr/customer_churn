import pandas as pd
import numpy as np

def preproccess(data_base):

    telco_data = data_base.copy()

    # Dropping unique ID value
    telco_data = telco_data.drop("customerID",axis = 1)    
 
    telco_data["TotalCharges"] = pd.to_numeric(telco_data["TotalCharges"],errors="coerce")
    telco_data.loc[telco_data["tenure"] == 0, "TotalCharges"] = 0
    telco_data["TotalCharges"] = telco_data["TotalCharges"].fillna(telco_data["TotalCharges"].median())

   # We fill those null values of TotalCharges with 0.

    telco_data["tenure_log"] = np.log1p(telco_data["tenure"])
    telco_data.drop("tenure",axis =1,inplace = True)


    telco_data["gender"] = telco_data["gender"].map({"Male" : 1,"Female":0})


    ord_cat_columns = ["Churn","Partner","Dependents","PhoneService","PaperlessBilling"]

    for col in ord_cat_columns:

        telco_data[col] = telco_data[col].map({"Yes":1,"No":0})

    telco_data_final = pd.get_dummies(telco_data,drop_first=True,dtype = int)

    return telco_data_final

