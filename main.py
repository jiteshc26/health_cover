from flask import Flask,render_template,request
import json
import pickle
import numpy as np
from app.utils import prediction
import CONFIG

app=Flask(__name__)

@app.route("/",methods = ["POST","GET"])
def start():
    return render_template("insurance.html")

@app.route("/predict",methods = ["POST","GET"])
def predict_price():
   data=request.form
   pred_obj = prediction()
   predicted_cover = pred_obj.predict_cover(data)
   print(predicted_cover)

   return render_template ("insurance.html",PREDICT_VALUE=predicted_cover)

if __name__ == "__main__":
    app.run(debug=True,host=CONFIG.HOST_NAME,port=CONFIG.PORT_NUMBER)