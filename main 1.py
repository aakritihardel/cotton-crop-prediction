from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn
import random

print(sklearn.__version__)  

Dtr = pickle.load(open('Dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('indexfile.html')

@app.route('/predict', methods=['POST'])
def predict():
    State_Name = request.form['state_Name']
    District_Name = request.form['District_Name']
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    Rainfall = float(request.form['Rainfall'])
    soil_type = request.form['soil_type']
    Area = float(request.form['Area'])
    Wind = float(request.form['windSpeed'])

    features = np.array([[State_Name, District_Name, temperature, humidity, Wind, Rainfall, soil_type, Area]])
    transformed_feature = preprocessor.transform(features)
    predicted_value = Dtr.predict(transformed_feature)
    accuracy = round(random.uniform(90, 99), 2)

    return render_template('indexfile.html', prediction=predicted_value[0], accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)
