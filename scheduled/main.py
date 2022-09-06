
import os
from tools.query import get_data_from_db, update_data_to_db
from iris import predict_iris


def main():
    iris_data=get_data_from_db()
    ids=[]
    input_data=[]
    for iris in iris_data:
        ids.append(iris[0])
        input_data.append([float(item) for item in iris[1:]])
    
    prediction=predict_iris(input_data)

    for idx in range(len(prediction)):
        update_data_to_db([ids[idx],prediction[idx][0]])
    

if __name__ == '__main__':
    main()