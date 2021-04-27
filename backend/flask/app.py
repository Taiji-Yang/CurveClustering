from flask import Flask, request, render_template, redirect
from high_dataset import read_from_frontend
from flask_sqlalchemy import SQLAlchemy
import os
from sqlite_database import *
import sqlite3

app = Flask(__name__)
tuples = None
cdata = get_data(['traction', 'aflow', 'module_num', 'time'])
feature = None

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///curves.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
#database_data = db.Table('curves', db.metadata, autoload = True, autoload_with = db.engine)
#print(db.session.query(database_data.Column.cell_pos_x).all())
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
    Curve_Id = db.Column('id', db.Integer, primary_key = True, autoincrement=False)
    X_Position = db.Column('cell_pos_x', db.String)
    Y_Position = db.Column('cell_pos_y', db.String)
    Time = db.Column('traction', db.String)
    Traction = db.Column('aflow', db.String)
    Aflow = db.Column('module_num', db.String)
    Module_Num = db.Column('time', db.String)

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
    global feature
    feature = request.json['required']
    return {}

@app.route('/databaseget', methods = ['GET'])
def databaseget():
    global feature
    data = None
    '''
    with sqlite3.connect('curves.db', check_same_thread=False) as con:
        cur = con.cursor()
        data = get_data(feature)
        con.commit()
    '''
    data = get_data(feature)
    print(data)
    return {
        "result": data
    }

@app.route('/resultplot', methods = ['GET'])
def resultplot():
    global tuples
    temp_traction = []
    temp_aflow = []
    temp_module_num = []
    temp_time = []
    for tuple_i in range(0, len(tuples)):
        startp = tuples[tuple_i][0]
        endp = tuples[tuple_i][1]
        temp_traction.append(cdata[tuple_i][0][startp:(endp+1)])
        temp_aflow.append(cdata[tuple_i][1][startp:(endp+1)])
        temp_module_num.append(cdata[tuple_i][2][startp:(endp+1)])
        temp_time.append(cdata[tuple_i][3][startp:(endp+1)])
    return {
        'traction': temp_traction,
        'aflow': temp_aflow,
        'module_num': temp_module_num,
        'time': temp_time,
    }

if __name__ == "__main__":
    #db.create_all()
    app.run(debug=True)