"""
This script  prepares subsets and train-test-val splits of the cleaned data.
"""
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

def main(args):
    # Load cleaned dataset from CSV
    df_clean = pd.read_csv(args.input_path)
    # ds_clean = Dataset.from_dict(ds_clean)
    # df_clean = pd.DataFrame(ds_clean["train"])
  
    # Create a column that denotes whether both the article and the reference summary passed quality checks
    df_clean["passed"] = (df_clean["passed_quality"] == True) & (df_clean["passed_quality_sum"] == True)

    # Create train, val, and test splits
    test_len = round(len(df_clean) * args.test_split)
    val_len = round(len(df_clean) * args.test_split)

    train, test = train_test_split(df_clean, test_size=test_len, random_state=args.seed)
    train, val = train_test_split(train, test_size=val_len, random_state=args.seed)

    # train, test = ds_clean.train_test_split(test_size=test_len, seed=args.seed).values()
    # train, val = train.train_test_split(test_size=val_len, seed=args.seed).values()
    #print(train.to_pandas().head())
    # Save train, val, and test datasets to CSV
    train.to_csv(args.train_output_path, index=False)
    val.to_csv(args.val_output_path, index=False)
    test.to_csv(args.test_output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and save a cleaned dataset.")
    parser.add_argument("--input_path", type=str, required=True,default='./tok_ds_clean_.csv' ,help="Path to the input CSV file.")
    parser.add_argument("--train_output_path", type=str, required=True,default='./train_all_.csv' ,help="Path to save the training CSV file.")
    parser.add_argument("--val_output_path", type=str, required=True,default = './val_abstractive_.csv' , help="Path to save the validation CSV file.")
    parser.add_argument("--test_output_path", type=str,default = './test_abstractive_.csv', required=True, help="Path to save the test CSV file.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Percentage of data to use for the test set.")
    parser.add_argument("--seed", type=int, default=22, help="Seed for reproducibility.")
    args = parser.parse_args()
    main(args)
