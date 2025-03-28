from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)

#loading models
Dtr=pickle.load(open('Dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))
#creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('indexfile.html')

@app.route('/predict', methods=['POST'])
def predict():
     
     State_Name = request.form['state_Name']
     District_Name = request.form['District_Name']
     temperature =  request.form['temperature']
     humidity = request.form['humidity']
     Rainfall = request.form['Rainfall']
     soil_type = request.form['soil_type']
     Area = request.form['Area']
     Wind = request.form['windSpeed']

     features = np.array([[State_Name, District_Name, temperature, humidity, Wind, Rainfall, soil_type, Area]])

     transformed_feature = preprocessor.transform(features)
     predicted_value = Dtr.predict(transformed_feature)

     return render_template('indexfile.html', prediction=predicted_value[0])

# python main
if __name__=="__main__":
     app.run(debug=True)