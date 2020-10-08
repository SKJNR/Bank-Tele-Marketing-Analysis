from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('bank.pickle', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods = ['POST'])
def predict():

    input_features = [x for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name=['age','marital','contact','education','poutcome','job','housing','loan','default','balance','month_int','date','duration','pdays','campaign','previous']
    df = pd.DataFrame(features_value, columns=features_name)
    prediction = model.predict(df)
    output = prediction[0]
    if output == 0:
       res_val ="Client Not Subscribed Term Deposit"
    else:
        res_val ='Oh Congrats ! Client Subscribed Term Deposit'
    return render_template('home.html', prediction_text=' {}'.format(res_val))


if __name__ == "__main__":
    app.run(debug=True)
