# Iris-Classification-Implemented
this is implementation of Iris Classification, created for completing an interview
## Training Model
this module can be executed using few arguments

arguments|type|description|default
:--------|:--:|:---------:|------:
--dataset|str|define which iris dataset is used, disabled if query is defined|https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv
--query|bool|use predefined query to get dataset|False
--batch|int|define batch size for training|20
--epoch|int|define number of epoch for training|10
--location|str|location of where to save model|./inferencing/models/
--modelname|str|define model name|model

***query feature is unfinished*

to execute training, just run *./training/train.py* file followed by modified arguments

example:
```
python3 ./training/train.py --epoch 20
```

## API inference

to run model in API inference, execute:

```
python3 ./inferencing/server.py
```

you can access the swagger of the api on:
```
http://localhost:5000/docs
```
curl example:
```
curl -X 'POST' \
  'http://localhost:5000/api/v1/iris_classification/prediction' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "identification": [
    {
      "sepal_length": 0,
      "sepal_width": 0,
      "petal_length": 0,
      "petal_width": 0
    }
  ]
}'
```

response:
```
{
  "result": [
    {
      "sepal_length": 0,
      "sepal_width": 0,
      "petal_length": 0,
      "petal_width": 0,
      "class_id": 0,
      "class_name": "setosa",
      "score": 0.4527132511138916
    }
  ]
}
```