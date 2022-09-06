import psycopg2
from configparser import ConfigParser

flower_name= ['setosa', 'versicolor', 'virginica']


def preprocess_input(input_raw):
    preprocessed_input=[]
    temp=[input_raw[key] for key in input_raw]
    return temp

def config_parser(filename='config.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            'Section {0} not found in the {1} file'.format(section, filename))

    return db


def execute_db(sql,additional=(),configsection=None):
    conn = None
    db_params=config_parser(section=configsection)

    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        cur.execute(sql,additional)
            
        conn.commit()
        cur.close()
    except psycopg2.DatabaseError as error:
        print(error)
        print('cannot connect to database')
        raise psycopg2.databaseError
    finally:
        if conn is not None:
            conn.close()

def retrieve_db(sql,additional=(),configsection=None, all=False):
    conn = None
    db_params=config_parser(section=configsection)

    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        cur.execute(sql,additional)
            
        if all:
            mview = cur.fetchall()
        else:
            mview = cur.fetchone()
        conn.commit()
        
        cur.close()
        return mview

    except psycopg2.DatabaseError as error:
        print(error)
        print('cannot connect to database')
        raise psycopg2.databaseError
    finally:
        if conn is not None:
            conn.close()
