import pickle
from flask import Flask, request


app = Flask(__name__)

with open('churn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict_churn', methods=['GET'])
def predict_churn():
    is_male = int(request.args.get('is_male'))
    num_inters = int(request.args.get('num_inters'))
    late_on_payment = int(request.args.get('late_on_payment'))
    age = int(request.args.get('age'))
    years_in_contract = float(request.args.get('years_in_contract'))

    prediction = model.predict([[is_male, num_inters, late_on_payment, age, years_in_contract]])[0]

    return str(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
