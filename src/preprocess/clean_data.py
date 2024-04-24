import argparse
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from ftfy import fix_text
from transformers import AutoTokenizer
from quality_filter import QualityFilter

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main(args):
    # Load JSON data
    ds = load_dataset("json", data_files={"train": args.dataset_path})
    ds_train = ds["train"]

    # Fix text and rename columns
    t_sums = [fix_text(i) for i in ds_train["aiGeneratedHeadline"]]
    t_texts = [fix_text(i) for i in ds_train["article"]]
    ds_train = ds_train.add_column("summary_fix", t_sums)
    ds_train = ds_train.add_column("text_fix", t_texts)
    ds_train = ds_train.remove_columns(["aiGeneratedHeadline", "article"])
    ds_train = ds_train.rename_column("summary_fix", "summary")
    ds_train = ds_train.rename_column("text_fix", "text")

    # Tokenize and preprocess
    print("Starting preprocessing")
    tok_dd = ds_train.map(preprocess_function, batched=True)
    tok_ds = tok_dd
    print("Done preprocessing")

    # Add features to ds with the tokenized lengths
    tok_text_len = [len(text) for text in tok_dd["input_ids"]] 
    tok_ds = tok_dd.add_column("tok_text_len", tok_text_len)
    tok_sum_len = [len(summary) for summary in tok_ds["labels"]]
    tok_ds = tok_ds.add_column("tok_sum_len", tok_sum_len)

    # Filtering based on cutoffs
    tok_df = pd.DataFrame(tok_ds)
    tok_df_clean = tok_df[
        (tok_df["tok_sum_len"] >= np.quantile(tok_ds["tok_sum_len"], 0.02)) &
        (tok_df["tok_sum_len"] <= np.quantile(tok_ds["tok_sum_len"], 0.98)) &
        (tok_df["tok_text_len"] >= np.quantile(tok_ds["tok_text_len"], 0.02)) &
        (tok_df["tok_text_len"] <= np.quantile(tok_ds["tok_text_len"], 0.98))
    ]

    tok_ds_clean = Dataset.from_pandas(tok_df_clean)

    # Apply filters
    qf = QualityFilter(
        min_stop_words=2,
        mean_word_length=(3, 10),
        doc_length=(10, 100_000),
        alpha_ratio=0.6,
        duplicate_lines_chr_fraction=0.4,
        duplicate_paragraph_chr_fraction=0.4,
        top_ngram_chr_fraction_thresholds=[0.20, 0.18, 0.16],
        top_ngram_chr_fraction_range=(2, 4),
        top_ngram_min_count=3,
        duplicate_n_gram_fraction_thresholds=[
            0.25, 0.24, 0.23, 0.22, 0.21, 0.20,
        ],
        ignore_filters=[
            "duplicate_ngram_chr_fraction",
            "top_ngram_chr_fraction",
            "line_bullets_or_ellipsis",
            "detect_language",
            "short_long_sentece",
        ],
    )
    filter_to_ignore = [
        "doc_length",
        "alpha_ratio",
        "symbol_2_word_ellipsis",
        "duplicate_lines_chr_fraction",
        "top_ngram_chr_fraction",
        "duplicate_ngram_chr_fraction",
        "detect_language",
        "stop_word",
        "mean_word_length",
        "line_bullets_or_ellipsis",
    ]
    qf_sum = QualityFilter(
        min_stop_words=2,
        mean_word_length=(3, 10),
        doc_length=(10, 100_000),
        alpha_ratio=0.6,
        duplicate_lines_chr_fraction=0.5,
        duplicate_paragraph_chr_fraction=0.6,
        top_ngram_chr_fraction_thresholds=[0.20, 0.18, 0.16],
        top_ngram_chr_fraction_range=(2, 4),
        top_ngram_min_count=3,
        duplicate_n_gram_fraction_thresholds=[
            0.25, 0.24, 0.23, 0.22, 0.21, 0.20,
        ],
        ignore_filters=filter_to_ignore,
    )

    print("Applying quality filters")
    texts = tok_ds_clean["text"]
    summaries = tok_ds_clean["summary"]
    filtered = qf.describe_filter(texts)
    filtered_sum = qf_sum.describe_filter(summaries)
    passed_quality = [None] * len(texts)
    filter = [None] * len(texts)
    passed_quality_sum = [None] * len(texts)
    filter_sum = [None] * len(texts)

    for n, i in enumerate(texts):
        print("Looping i:", i)
        result = next(filtered)
        result_sum = next(filtered_sum)
        if result == "passed filters":
            passed_quality[n] = True
            filter[n] = "nan"
        else:
            passed_quality[n] = False
            filter[n] = result

        if result_sum == "passed filters":
            passed_quality_sum[n] = True
            filter_sum[n] = "nan"
        else:
            passed_quality_sum[n] = False
            filter_sum[n] = result_sum

    tok_ds_clean = tok_ds_clean.add_column("passed_quality", passed_quality)
    tok_ds_clean = tok_ds_clean.add_column("filter", filter)
    tok_ds_clean = tok_ds_clean.add_column("passed_quality_sum", passed_quality_sum)
    tok_ds_clean = tok_ds_clean.add_column("filter_sum", filter_sum)

    # Save cleaned data to CSV
    tok_ds_clean.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    print("Starting cleaning process")
    parser = argparse.ArgumentParser(description="Clean and filter a JSON dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the input JSON dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the cleaned CSV file.")
    args = parser.parse_args()
    main(args)
