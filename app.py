import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))  

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print("Received data:", data)
    
    # Convert JSON to numpy array
    input_array = np.array(list(data.values())).reshape(1, -1)
        #we getting thr values in key value pairs so take the values and reshape it because to predict it(as it accepts the 2d array)

    # Transform using the same scaler used during training
    transformed_data = scaler.transform(input_array)
    
    # Make prediction
    output = regmodel.predict(transformed_data)[0]
    print("Prediction:", output)
    
    return jsonify({'prediction': float(output)})  


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    #this we get the form values that we entire in html
   # final_ip=scaler.transform(np.array(data).reshape(-1,1))
    final_ip = scaler.transform([data])
    print(final_ip)
    output=regmodel.predict(final_ip)[0]
    return render_template("home.html",prediction_text="The house price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
