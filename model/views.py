from django.shortcuts import render
import joblib
import pandas as pd
import numpy as np


def home(request):
    context={}
    return render(request, 'model/home.html', context=context)
    
    
def result(request):
    model=joblib.load('utils/logistic_model.pkl')
    feature_col = ['Property_Area','Education', 'Dependents']
    train_features = ['Credit_History', 'Education', 'Gender']
    
    Data=request.POST
    data=np.array([list(dict(Data).values())[1:]]).reshape(1,-1)
    col=list(dict(Data).keys())[1:]
    loan_test = pd.DataFrame(data, columns=col)
    
    loan_test.Gender = loan_test.Gender.replace({"Male": 1, "Female" : 0})
    loan_test.Married = loan_test.Married.replace({"Yes": 1, "No" : 0})
    loan_test.Self_Employed = loan_test.Self_Employed.replace({"Yes": 1, "No" : 0})
    
    
    le = joblib.load('utils/logistic_label_encoder.pkl')
    for col in feature_col:
        loan_test[col] = le.fit_transform(loan_test[col])
        
    x_test = loan_test[train_features].values
    predicted = model.predict(x_test)
    
    status=['Satisfied' if i==1  else 'Rejected' for i in predicted]
    
    context={'condition': status[0]}
    return render(request, 'model/result.html', context=context)