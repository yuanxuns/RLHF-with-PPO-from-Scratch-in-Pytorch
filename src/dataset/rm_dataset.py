from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Dict


def convert_to_reward_model_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_len: int,
    max_size_per_category: int = None,
) -> Dataset:
    """
    Convert a dataset for reward model training into the format required by the model.

    The input dataset should have two columns: "text" and "label". The "text" column should contain the text samples,
    and the "label" column should contain the label for each sample (0 or 1). The output dataset will have four columns:
    "rejected_input_ids", "rejected_attention_mask", "chosen_input_ids", and "chosen_attention_mask". The first two
    columns contain the tokenized and encoded rejected samples, and the last two columns contain the tokenized and
    encoded chosen samples.

    Args:
        dataset: The input dataset to convert.
        tokenizer: The tokenizer to use for tokenization.
        max_len: The maximum length of the input text samples.
        max_size_per_category: The maximum number of samples per category (rejected or chosen) to include in the
            output dataset. If None, all samples are included.

    Returns:
        A new dataset in the format required by the reward model.
    """
    df = dataset.to_pandas()
    negative_df = df[df["label"] == 0]
    positive_df = df[df["label"] == 1]
    negative_df = negative_df.drop(columns=["label"]).rename(
        columns={"text": "rejected"}
    )
    positive_df = (
        positive_df.sample(frac=1, random_state=0)
        .reset_index(drop=True)
        .drop(columns=["label"])
        .rename(columns={"text": "chosen"})
    )
    if max_size_per_category is not None:
        negative_df = negative_df.sample(
            n=min(len(negative_df), max_size_per_category), random_state=0
        ).reset_index(drop=True)
        positive_df = positive_df.sample(
            n=min(len(positive_df), max_size_per_category), random_state=0
        ).reset_index(drop=True)
    joined_df = negative_df.join(positive_df)

    rejected_encoded = tokenizer(
        joined_df["rejected"].tolist(),
        padding="max_length",
        max_length=max_len,
        truncation=True,
        add_special_tokens=False,
    )
    chosen_encoded = tokenizer(
        joined_df["chosen"].tolist(),
        padding="max_length",
        max_length=max_len,
        truncation=True,
        add_special_tokens=False,
    )
    joined_df["rejected_input_ids"] = rejected_encoded["input_ids"]
    joined_df["rejected_attention_mask"] = rejected_encoded["attention_mask"]
    joined_df["chosen_input_ids"] = chosen_encoded["input_ids"]
    joined_df["chosen_attention_mask"] = chosen_encoded["attention_mask"]

    return Dataset.from_pandas(joined_df, preserve_index=False).with_format("torch")


def build_dataloader(tokenizer: AutoTokenizer, config) -> Dict[str, DataLoader]:
    """
    Build PyTorch DataLoaders for reward model training and evaluation.

    This function loads the reward model dataset, converts it into the format required
    for reward model training and evaluation using the provided tokenizer, and wraps
    the datasets in PyTorch DataLoaders.

    Args:
        tokenizer: An instance of AutoTokenizer used for tokenizing the dataset.
        config: A configuration object containing settings for loading and processing
                the dataset, including paths, batch sizes, and maximum lengths.

    Returns:
        A dictionary containing two DataLoaders:
        - "train_dataloader": DataLoader for the training dataset.
        - "test_dataloader": DataLoader for the evaluation dataset.
    """

    dataset = load_dataset(config["data"]["rm_dataset"])
    del dataset["unsupervised"]

    train_dataset = convert_to_reward_model_dataset(
        dataset["train"], tokenizer, config["training"]["rm"]["max_len"]
    )
    test_dataset = convert_to_reward_model_dataset(
        dataset["test"],
        tokenizer,
        config["training"]["rm"]["max_len"],
        config["training"]["rm"]["max_eval_size_per_category"],
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=config["training"]["rm"]["batch_size"]
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["training"]["rm"]["eval_batch_size"]
    )

    return {"train_dataloader": train_dataloader, "test_dataloader": test_dataloader}
