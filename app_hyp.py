from flask import*
import numpy as np
import pickle
model = pickle.load(open('hyp.pkl', 'rb'))
app = Flask(__name__)
@app.route("/")
def home():
    return render_template('input_1.html')
@app.route("/predict",methods=['post'])
def predict():
    data1 = request.form['Age']
    data2 = request.form['Sex']
    data3 = request.form['on_thyroxine']
    data4 = request.form['query_on_thyroxine']
    data5 = request.form['on_antithyroid_medication']
    data6 = request.form['thyroid_surgery']
    data7 = request.form['query_hypothyroid']
    data8 = request.form['query_hyperthyroid']
    data9 = request.form['pregnant']
    data10 = request.form['sick']
    data11 = request.form['tumor']
    data12 = request.form['lithium']
    data13 = request.form['goitre']
    data14 = request.form['TSH_measured']
    data15 = request.form['TSH']
    data16 = request.form['T3_measured']
    data17 = request.form['T3']
    data18 = request.form['TT4_measured']
    data19 = request.form['TT4']
    data20 = request.form['T4U_measured']
    data21 = request.form['T4U']
    data22 = request.form['FTI_measured']
    data23 = request.form['FTI']
    data24 = request.form['TBG_measured']
    data25 = request.form['TBG']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9, data10, data11, data12,data13,data14,data15,data16,data17,data18,data19, data20, data21, data22,data23,data24,data25]])
    pred = model.predict(arr)
    return render_template('prediction.html', data=pred)
    #return model
if __name__ == "__main__":
    app.run(debug = True)