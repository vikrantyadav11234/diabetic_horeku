from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['post'])
def predict():
    
    
    # load the model
    model = joblib.load('diebetic_80.pkl')
    preg = request.form.get('Preg')
    plas = request.form.get('Plas')
    pres = request.form.get('Pres')
    skin = request.form.get('Skin')
    test = request.form.get('Test')
    mass = request.form.get('Mass')
    pedi = request.form.get('Pedi')
    age = request.form.get('Age')
     
    print(preg,plas,pres, skin, test, mass, pedi, age)
    array = np.array([[preg,plas,pres,skin,test,mass,pedi,age]])
    array = np.array(array, dtype= float)
    output = model.predict(array)
    print(output)
    if output[0] == 0:
        data = "Person is diabetic"
    else:
        data= "Person is not diabetic"
    return render_template('predict.html', data = data)
if __name__ == "__main__":
    app.run(debug = True)