from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app=Flask(__name__)

classifier=pickle.load(open("Diabetes-Prediction/diabetes_model.pkl","rb"))
scaler=pickle.load(open("Diabetes-Prediction/diabetes_model1.pkl","rb"))


@app.route('/')

@app.route('/diabetes')
def home():
    return render_template("diabetes.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method=='POST':
        Pregnancies=int(request.form["Pregnancies"])
        Glucose=float(request.form["Glucose"])
        BloodPressure=int(request.form["BloodPressure"])
        SkinThickness=float(request.form["SkinThickness"])
        Insulin=float(request.form["Insulin"])
        BMI=float(request.form["BMI"])
        DiabetesPedigreeFunction=float(request.form["DiabetesPedigreeFunction"])
        Age=int(request.form["Age"])

        input_data=(Pregnancies,Glucose, BloodPressure, SkinThickness,Insulin, BMI, DiabetesPedigreeFunction, Age)
        input_data_as_numpy_array=np.asarray(input_data)
        input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
        std_data = scaler.transform(input_data_reshaped)
        prediction=classifier.predict(std_data)

        if(prediction[0]==1):
            result="Sorry, you have chances of getting the disease. Please consult the doctor immediately."
        else:
            result="No need to fear. You have no dangerous symptoms of the disease."

        return render_template("result.html", result=result)
    

if __name__=='__main__':
    app.run(debug=True)
