
from flask import Flask, request, jsonify, render_template
import pickle
import os
import json
import sklearn

# Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


filename = "linier_regression_dbp_model_fix2.sav"
filename2 = "linier_regression_sbp_model_fix2.sav"

model_dbp = pickle.load(open(filename, 'rb'))
model_sbp = pickle.load(open(filename2, 'rb'))

print('The scikit-learn version is {}.'.format(sklearn.__version__))
# Predict
@app.route('/', methods=['POST'])
def predict():

    
    sex   = request.form['sex']
    age = request.form['age']
    olahraga = request.form['olahraga']
    tobacco = request.form['tobacco']
    sbp = request.form['sbp']
    dbp = request.form['dbp']
    imt = request.form['imt']
    # hbp_stat = request.form['hbp_stat']
    domisili = request.form['domisili']
    tobacco_long_rec2 = request.form['tobacco_long_rec2']
    work = request.form['work']
    education = request.form['education']
    
   
    hasil = []

    for x in range(0,10,1):
        age = int(age)+1
        input_sbp = [[sex,age,olahraga,tobacco,dbp,imt,domisili,tobacco_long_rec2,work,education]]
        input_dbp = [[sex,age,olahraga,tobacco,sbp,imt,domisili,tobacco_long_rec2,work,education]]
        # input_sbp = [[weight,sex,age,olahraga,tobacco,sbp,dbp,tobacco_long_rec2,domisili]]
        output_sbp = model_sbp.predict(input_sbp)
        output_dbp = model_dbp.predict(input_dbp)
        # output_sbp = model_sbp.predict(input_sbp)

        hasil.append([age,output_sbp[0],output_dbp[0]])

    return str(hasil)

@app.route('/home')
def home():
    return render_template('index.html')


# Run Server
if __name__ == '__main__':
    app.run(debug=True)
