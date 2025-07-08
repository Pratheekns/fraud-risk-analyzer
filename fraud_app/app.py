from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and features
model = joblib.load('fraud_model.pkl')
features = pd.read_csv('company_features.csv', index_col='company_id')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    company_id = request.form['company_id']
    if company_id not in features.index:
        return render_template('result.html', company_id=company_id, level="Not Found", note="❌ Invalid Company ID", color="gray")

    input_data = features.loc[company_id].values.reshape(1, -1)
    risk = model.predict(input_data)[0]

    # Return status
    if risk == 0:
        return render_template('result.html', company_id=company_id, level="Low", note="🟢 Free from risk", color="green")
    elif risk == 1:
        return render_template('result.html', company_id=company_id, level="Medium", note="🟡 Be cautious", color="orange")
    else:
        return render_template('result.html', company_id=company_id, level="High", note="🔴 Immediate action required", color="red")

if __name__ == '__main__':
    app.run(debug=True)
