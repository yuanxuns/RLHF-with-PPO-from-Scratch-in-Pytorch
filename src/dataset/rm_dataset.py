from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import Dict

def convert_to_reward_model_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_len: int) -> Dataset:
  df = dataset.to_pandas()
  negative_df = df[df['label']==0]
  positive_df = df[df['label']==1]
  negative_df = negative_df.drop(
      columns=['label']).rename(
      columns={'text': 'rejected'})
  positive_df = positive_df.sample(
      frac=1, random_state=0).reset_index(
      drop=True).drop(columns=['label']).rename(
      columns={'text': 'chosen'})
  joined_df = negative_df.join(positive_df)
  
  rejected_encoded = tokenizer(joined_df['rejected'].tolist(),
                               padding='max_length',
                               max_length=max_len,
                               truncation=True,
                               add_special_tokens=False)
  chosen_encoded = tokenizer(joined_df['chosen'].tolist(),
                             padding='max_length',
                             max_length=max_len,
                             truncation=True,
                             add_special_tokens=False)
  joined_df['rejected_input_ids'] = rejected_encoded['input_ids']
  joined_df['rejected_attention_mask'] = rejected_encoded['attention_mask']
  joined_df['chosen_input_ids'] = chosen_encoded['input_ids']
  joined_df['chosen_attention_mask'] = chosen_encoded['attention_mask']
  
  return Dataset.from_pandas(joined_df, preserve_index=False).with_format("torch")

def build_dataloader(tokenizer: AutoTokenizer, config) -> Dict[str, DataLoader]:
  dataset = load_dataset(config["data"]["rm_dataset"])
  del dataset['unsupervised']
  
  train_dataset = convert_to_reward_model_dataset(dataset["train"], tokenizer, config["training"]["rm"]["max_len"])
  test_dataset = convert_to_reward_model_dataset(dataset["test"], tokenizer, config["training"]["rm"]["max_len"])
  
  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config["training"]["rm"]["batch_size"])
  test_dataloader = DataLoader(test_dataset, batch_size=config["training"]["rm"]["eval_batch_size"])
  
  return {"train_dataloader": train_dataloader, "test_dataloader": test_dataloader}
  