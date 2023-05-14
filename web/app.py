from flask import Flask, render_template, request, url_for
import pickle
import numpy as np 
import os
from datetime import datetime

app = Flask(__name__)
env_config = os.getenv("PROD_APP_SETTINGS", "config.DevelopmentConfig")
app.config.from_object(env_config)
model = pickle.load(open('saved_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    # print("AA")
    int_features = request.form.get('Date')
    newDate = datetime.strptime(int_features, '%Y-%m-%dT%H:%M')

    print(newDate)
    print(newDate.day)
    
    future = model.make_future_dataframe(periods=24000 ,freq='h')
    forecast = model.predict(future)
    date = datetime(newDate.year, newDate.month, newDate.day, newDate.hour, 0, 0)
    
    var =0
    if date>datetime(2017,5, 14,0,0,0):
        return render_template('test.html', prediction_text='The number of vehicles predicted is has not been predicted yet.')
    else:
        var = forecast.loc[forecast['ds']==date,'yhat'].values[0]
        var = var.round()
    

    return render_template('test.html', prediction_text='The number of vehicles predicted is {}'.format(var))


if __name__ == "__main__":
    app.run(debug=True)
    