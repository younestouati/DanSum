## Installations 

The Notebook contains the packages need for running the project on colab.

## Data

The notebook contains downloading the data into colab.

### Processing the data

The data should be processed using two scripts `clean_data.py` and `prepare_data_splits.py`

#### example for using with the arguments:

```python clean_data.py --dataset_path /content/unbait-dataset-full.json --output_path /content/data/tok_ds_clean_.csv```

and 

```python prepare_data_splits.py --input_path /content/data/tok_ds_clean_.csv --train_output_path /content/data/train_all_.csv --val_output_path /content/data/val_abstractive_.csv --test_output_path /content/data/test_abstractive_.csv```

once you do this process you can save the data and use it directly in the training scripts.

## Training 

training should be done using `train.py` the script argument are in config file

the most important arguments are: 

- model_checkpoint: "google/mt5-small" # str: e.g. "google/mt5-base" or "google/mt5-large" or "google/mt5-xl" or "google/mt5-xxl"
- train_path: "/content/drive/MyDrive/models/danish/data/train_all_.csv" # str
- val_path: "/content/drive/MyDrive/models/danish/data/val_abstractive_.csv" # str
- test_path: "/content/drive/MyDrive/models/danish/data/test_abstractive_.csv" # str
- quantiz: False #wether to have a quantized training or not




