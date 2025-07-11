import torch
from tqdm import tqdm

def  run_eval(model, test_dataloader, device, tb_writer, step, batch_size):
  
  total_correct_samples = 0
  with torch.no_grad():
      batch_iterator = tqdm(test_dataloader, desc="Eval", leave=False)
      for batch in batch_iterator:
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