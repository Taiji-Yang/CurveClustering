from high_dataset import *
import sqlite3
#conn = sqlite3.connect(':memory:')
conn = sqlite3.connect('curves.db', check_same_thread=False)
cursor = conn.cursor()

def insert_database(curve, id):
    value = [id] + curve
    cursor.execute("INSERT INTO curves VALUES (?,?,?,?,?,?,?)", value)
    conn.commit()

def database_create():
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS curves (
            id INTEGER,
            cell_pos_x TEXT, 
            cell_pos_y TEXT, 
            traction TEXT, 
            aflow TEXT, 
            module_num TEXT, 
            time TEXT
        )
        """
    )
    init_database = False

def curve_to_string(curve):
    res = []
    for dim in curve:
        res.append(','.join(map(str, dim)))
    return res

def fetch_values(feature):
    value = ','.join(feature)
    comm = "SELECT " + value + " FROM curves"
    cursor.execute(comm)
    conn.commit()
    return cursor.fetchall()

def get_data(feature):
    raw_data = fetch_values(feature)
    curves = []
    for single_curve in raw_data:
        curve = []
        for dim in single_curve:
            str_dim = dim.split(',')
            curve.append(list(map(float, str_dim)))
        curves.append(curve)
    return curves



if __name__ == "__main__":
    curves = request_curves(['cell_pos_x', 'cell_pos_y', 'traction', 'aflow', 'module_num', 'time'])
    database_create()
    id = 0
    for curve in curves:
        str_curve = curve_to_string(curve)
        insert_database(str_curve, str(id))
        id += 1
