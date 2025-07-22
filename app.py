from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


model = pickle.load(open('xgb_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        features = [request.form.get(feature) for feature in 
                    ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                     'ejection_fraction', 'high_blood_pressure', 'platelets', 
                     'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]
        
        
        if None in features or '' in features:
            return render_template('index.html', prediction_text="Please fill out all fields.")

        
        features = [float(f) for f in features]
        
       
        features = np.array(features).reshape(1, -1)

        
        prediction = model.predict(features)
        result = 'Heart Failure' if prediction[0] == 1 else 'No Heart Failure'
        return render_template('index.html', prediction_text=f'Prediction: {result}')
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
