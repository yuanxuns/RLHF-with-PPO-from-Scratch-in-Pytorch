import torch
from tqdm import tqdm
import gc 

def run_rm_eval(model, test_dataloader, device, tb_writer, step, batch_size):
  
  total_correct_samples = 0
  with torch.no_grad():
      batch_iterator = tqdm(test_dataloader, desc="Eval", leave=False)
      for batch in batch_iterator:
          gc.collect()
          torch.cuda.empty_cache()
          
          chosen_input_ids = batch['chosen_input_ids'].to(device)
          chosen_attention_mask = batch['chosen_attention_mask'].to(device)
          rejected_input_ids = batch['rejected_input_ids'].to(device)
          rejected_attention_mask = batch['rejected_attention_mask'].to(device)

          r_w = model(chosen_input_ids, attention_mask=chosen_attention_mask).logits
          r_l = model(rejected_input_ids, attention_mask=rejected_attention_mask).logits   
          
          batch_correct_samples = (r_w - r_l > 0).sum().item()
          total_correct_samples += batch_correct_samples
          batch_iterator.set_postfix({"correct_samples/batch_size": f"{batch_correct_samples:3d}/{batch_size:3d}"})
          
  tb_writer.add_scalar("eval/accuracy", total_correct_samples / (len(test_dataloader.dataset)*batch_size), step)

def run_ppo_eval(model, reward_model, val_dataloader, ppo_tokenizer, rm_tokenizer, generation_kwargs, tb_writer, device, step):
    with torch.no_grad():
      reward_model.to(device)
      rewards_sum = 0.0
      rewards_count = 0
      for batch in val_dataloader:
        query_ids = torch.nn.utils.rnn.pad_sequence(
            batch['input_ids'],
            batch_first=True,
            padding_value=ppo_tokenizer.pad_token_id,
        ).to(device)

        query_attention_masks = torch.nn.utils.rnn.pad_sequence(
            batch['attention_mask'],
            batch_first=True,
            padding_value=ppo_tokenizer.pad_token_id,
        ).to(device)

        query_response_ids = model.generate(
            input_ids=query_ids,
            attention_mask=query_attention_masks,
            **generation_kwargs
        )

        # response_ids = query_response_ids[:, query_ids.shape[1]:]
        attention_mask = query_response_ids.not_equal(ppo_tokenizer.pad_token_id).long()
        query_response_texts = ppo_tokenizer.batch_decode(query_response_ids, skip_special_tokens=True) 
        rm_encoded_query_response = rm_tokenizer(
            query_response_texts,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )
        
        scores = torch.sigmoid(reward_model(rm_encoded_query_response["input_ids"].to(device), rm_encoded_query_response["attention_mask"].to(device)).logits.squeeze(1))
        scores = 2 * (scores - 0.5)
        rewards_sum += scores.sum().item()
        rewards_count += len(scores)        
      reward_model.cpu()
      tb_writer.add_scalar("eval_avg_rewards", rewards_sum/rewards_count, step)
