
from .utils import execute_db,retrieve_db

def update_data_to_db(data):
    sql_dump_data = '''
        insert into 
            IrisOutput(executed_at, id, class) 
        values 
            (now(),%s,%s)
    '''

    execute_db(
            sql_dump_data,
            (
                data[0],
                data[1],
            ),"postgresql"
        )


def get_data_from_db():
    sql_get_data = '''
            select id, sepal_length, sepal_width, petal_length, petal_width
            from IrisInput
    '''
    data_tuple=retrieve_db(sql_get_data,additional=(),
                configsection='postgresql', 
                all=True
                )
    return data_tuple