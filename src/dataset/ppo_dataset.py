from datasets import load_dataset
from torch.utils.data import DataLoader


def tokenize(sample, tokenizer, num_input_tokens: int):
    """
    Tokenize a sample in the PPO dataset.

    :param sample: Sample from the PPO dataset
    :param tokenizer: Tokenizer to use for tokenization
    :param num_input_tokens: Maximum number of input tokens to keep
    :returns: Sample with "input_ids", "attention_mask", and "query" fields added
    """

    sample["input_ids"] = tokenizer.encode(sample["text"])[:num_input_tokens]
    sample["attention_mask"] = [1] * len(sample["input_ids"])
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample


def collator(batch):
    """
    Collate a batch of samples into a dictionary with lists of values for each key.

    :param batch: List of samples, where each sample is a dictionary.
    :returns: Dictionary where each key corresponds to a list of values from the batch samples.
    """

    return dict((key, [d[key] for d in batch]) for key in batch[0])


def build_dataloader(ppo_tokenizer, config):
    """
    Build a PyTorch DataLoader for the PPO dataset.

    :param ppo_tokenizer: Tokenizer to use for tokenizing the PPO dataset
    :param config: Configuration object with settings for building the DataLoader
    :returns: A dictionary with two DataLoaders: "train_dataloader" and "val_dataloader"
    """
    dataset = load_dataset(config["data"]["ppo_dataset"])
    del dataset["unsupervised"]
    ds_train, ds_val = dataset["train"], dataset["test"]
    ds_train = ds_train.filter(
        lambda x: len(x["text"].split(" ")) > config["training"]["ppo"]["min_num_words"]
    )
    ds_val = ds_val.filter(
        lambda x: len(x["text"].split(" ")) > config["training"]["ppo"]["min_num_words"]
    )

    # Limit validation set to a maximum number of samples
    max_val_samples = config["training"]["ppo"]["max_eval_size"]
    if max_val_samples is not None:
        ds_val = ds_val.select(range(min(max_val_samples, len(ds_val))))
    print(f"Validation dataset size: {len(ds_val)}")

    map_kwargs = {
        "batched": False,
        "remove_columns": ["text", "label"],
        "fn_kwargs": {
            "tokenizer": ppo_tokenizer,
            "num_input_tokens": config["training"]["ppo"]["num_input_tokens"],
        },
    }
    tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)
    tokenized_dataset_val = ds_val.map(tokenize, **map_kwargs)
    tokenized_dataset_train.set_format(type="torch")
    tokenized_dataset_val.set_format(type="torch")

    train_dataloader = DataLoader(
        tokenized_dataset_train,
        batch_size=config["training"]["ppo"]["batch_size"],
        collate_fn=collator,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        tokenized_dataset_val,
        batch_size=config["training"]["ppo"]["eval_batch_size"],
        collate_fn=collator,
        shuffle=True,
    )
    return {"train_dataloader": train_dataloader, "val_dataloader": val_dataloader}
