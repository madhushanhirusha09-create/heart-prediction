from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler (loads ONCE when server starts)
model = joblib.load('heart_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

# Valid ranges for every input field: (min, max, label shown in error message)
RANGES = {
    'age'      : (20, 100,  'Age'),
    'sex'      : (0, 1,     'Sex'),
    'cp'       : (0, 3,     'Chest Pain Type'),
    'trestbps' : (80, 220,  'Resting Blood Pressure'),
    'chol'     : (100, 600, 'Cholesterol'),
    'fbs'      : (0, 1,     'Fasting Blood Sugar'),
    'restecg'  : (0, 2,     'Resting ECG'),
    'thalach'  : (60, 220,  'Max Heart Rate'),
    'exang'    : (0, 1,     'Exercise Induced Angina'),
    'oldpeak'  : (0.0, 7.0, 'ST Depression'),
    'slope'    : (0, 2,     'Slope'),
    'ca'       : (0, 4,     'Major Vessels'),
    'thal'     : (0, 3,     'Thalassemia'),
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Validate every form value BEFORE predicting
    values = {}
    for field, (low, high, label) in RANGES.items():
        raw = request.form.get(field, '').strip()
        try:
            value = float(raw)
        except ValueError:
            return render_template('index.html',
                                   error=f'❌ {label}: "{raw}" is not a valid number.')
        if not (low <= value <= high):
            return render_template('index.html',
                                   error=f'❌ {label} must be between {low} and {high} (got {raw}).')
        values[field] = [value]

    patient = pd.DataFrame(values)

    # Apply the SAME scaling used in training
    patient_scaled = scaler.transform(patient)

    # Predict
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]

    if prediction == 1:
        result = f'⚠️ HEART DISEASE DETECTED ({probability[1]*100:.1f}% confidence)'
    else:
        result = f'✅ NO HEART DISEASE ({probability[0]*100:.1f}% confidence)'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
