from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('breast_cancer_model.h5')

# Recreate the scaler object
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = []
    for feature in request.form.values():
        input_data.append(float(feature))
    
    # Convert input data to numpy array and reshape it
    input_data = np.asarray(input_data)
    input_data = input_data.reshape(1, -1)
    
    # Standardize the input data
    input_data_std = scaler.fit_transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_std)
    
    # Convert prediction to label
    if prediction[0][0] > prediction[0][1]:
        result = 'Malignant'
    else:
        result = 'Benign'
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
