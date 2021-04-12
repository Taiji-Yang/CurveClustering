from flask import Flask, request, render_template, redirect
from high_dataset import read_from_frontend
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
tuples = None
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///curves.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
#curr_dir = os.path.abspath(os.path.dirname(__file__))
'''
@app.route('/', methods = ['POST', 'GET'])
def index():
    
    global tuples
    if request.method == 'POST':
        json_data = request.json
        tuples = read_from_frontend(json_data["input"], json_data["curveid"])
        return redirect('/')
    else:
        return render_template('index.html')
'''
'''
class curves(db.Model):
    Curve_Id = db.Column('Curve_Id', db.Integer, primary_key = True, autoincrement=False)
    X_Position = db.Column('X_Position', db.ARRAY(db.Float))
    Y_Position = db.Column('Y_Position', db.ARRAY(db.Float))
    Time = db.Column('Time', db.ARRAY(db.Integer))
    Traction = db.Column('Traction', db.ARRAY(db.Float))
    Aflow = db.Column('Aflow', db.ARRAY(db.Float))
    Module_Num = db.Column('Module_Num', db.ARRAY(db.Integer))

    def __init__(self, Curve_Id,  X_Position, Y_Position, Time, Traction, Aflow, Module_Num):
        self.Curve_Id = Curve_Id
        self.X_Position = X_Position
        self.Y_Position = Y_Position
        self.Time = Time
        self.Traction = Traction
        self.Aflow = Aflow
        self.Module_Num = Module_Num
'''


@app.route('/api', methods = ['POST'])
def api():
    global tuples
    json_data = request.json
    tuples = read_from_frontend(json_data["input"], json_data["curveid"])
    return {'status':'ok'}

@app.route('/result', methods = ['GET'])
def result():
    global tuples
    print('back')
    print(tuples)
    return {
        "results": tuples
    }

@app.route('/deleteall', methods = ['GET'])
def deleteall():
    global tuples
    tuples = []
    return {}

@app.route('/databasepost', methods = ['POST'])
def databasepost():
    features_required = request.json['required']
    return {}

if __name__ == "__main__":
    #db.create_all()
    app.run(debug=True)