from src.models.utils import get_device, seed_everything, DTYPE_MAP
from src.tokenizer.tokenizer import get_tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from torch import nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from copy import deepcopy
from pathlib import Path
import sys
from src.models.memory_efficient_adam import MemoryEfficientAdamW
from tqdm import tqdm
import gc
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

def tokenize(sample, tokenizer, num_input_tokens: int):
    sample['input_ids'] = tokenizer.encode(sample['text'])[:num_input_tokens]
    sample['attention_mask'] = [1] * len(sample['input_ids'])
    sample['query'] = tokenizer.decode(sample['input_ids'])
    return sample
  
def collator(batch):
    return dict((key, [d[key] for d in batch]) for key in batch[0])   
   
def build_dataloader(ppo_tokenizer, config):
  dataset = load_dataset(config["data"]["ppo_dataset"])
  del dataset['unsupervised']  
  ds_train, ds_val = dataset['train'], dataset['test']
  ds_train = ds_train.filter(lambda x: len(x['text'].split(' ')) > config["training"]["ppo"]["min_num_words"])
  ds_val = ds_val.filter(lambda x: len(x['text'].split(' ')) > config["training"]["ppo"]["min_num_words"])

  map_kwargs = {
      "batched": False,
      "remove_columns": ['text', 'label'],
      "fn_kwargs": {
        "tokenizer": ppo_tokenizer,        
        "num_input_tokens": config["training"]["ppo"]["num_input_tokens"],
      }    
  }
  tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs)
  tokenized_dataset_val = ds_val.map(tokenize, **map_kwargs)
  tokenized_dataset_train.set_format(type='torch')
  tokenized_dataset_val.set_format(type='torch') 
  
  batch_size = config["training"]["ppo"]["batch_size"]
  train_dataloader = DataLoader(tokenized_dataset_train, batch_size=batch_size, collate_fn=collator, shuffle=True)
  val_dataloader = DataLoader(tokenized_dataset_val, batch_size=batch_size, collate_fn=collator, shuffle=True)
  return {"train_dataloader": train_dataloader,
          "val_dataloader": val_dataloader}


class ValueHead(nn.Module):
  def __init__(self, llm_config, device):
    super().__init__()
    self.hidden_size = llm_config.hidden_size
    num_labels = 1
    self.value_head = nn.Sequential(
      nn.LayerNorm(self.hidden_size),
      nn.GELU(),
      nn.Linear(self.hidden_size, 4 * self.hidden_size),
      nn.GELU(),
      nn.Linear(4*self.hidden_size, num_labels),
    ).to(llm_config.torch_dtype).to(device)
    
    for layer in self.value_head:
      if isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm):
        nn.init.normal_(layer.weight, std=(1.0 / np.sqrt(np.max(layer.weight.size()) + 1)) )
        nn.init.zeros_(layer.bias)
        
  def forward(self, hidden_states):
    return self.value_head(hidden_states)
    
class ModelForCausalLMWithValueHead(nn.Module):
  def __init__(self, model_path, torch_dtype):
    super().__init__()
    self.llm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    self.value_head = ValueHead(self.llm.config, self.llm.device)
    
  def forward(self, input_ids, attention_mask):
    llm_outputs = self.llm.forward(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    llm_logits = llm_outputs.logits
    last_hidden_state = llm_outputs.hidden_states[-1]
    values = self.value_head(last_hidden_state).squeeze(-1)
    return llm_logits, values #(B, seq_len, vocab_size), (B)
  
  def generate(self, *args, **kwargs):
      return self.llm.generate(*args, **kwargs)
    
    
def get_preloaded_models_and_optimizer(config, dtype, device):
  rm_file_path = Path(config["training"]["ppo"]["pretrained_reward_model_path"])
  if not rm_file_path.exists():
    print("Error: Pretrained reward model file is missing.")
    sys.exit(0)
  
  reward_model = AutoModelForSequenceClassification.from_pretrained(
    config["training"]["ppo"]["pretrained_reward_model_path"],
    torch_dtype=dtype,
    num_labels=1,
  ).cpu().eval()
  
  model = ModelForCausalLMWithValueHead(
            config["training"]["ppo"]["tokenizer_name_or_path"],
            torch_dtype=dtype,
        ).to(device)
  
  optimizer = MemoryEfficientAdamW(
    model.parameters(),
    lr=float(config["training"]["ppo"]["learning_rate"]),
    weight_decay=config["training"]["ppo"]["weight_decay"],
    betas=config["training"]["ppo"]["betas"],
    enabled=config["training"]["ppo"]["enable_memory_efficient_adamw"],
  )
  
  step = 1
  num_epoch = 1
  file_path = Path(config["training"]["ppo"]["states_file"])
  if file_path.exists():
    states = torch.load(config["training"]["ppo"]["states_file"])
    step = states["step"]
    num_epoch = states["num_epoch"]
    model.load_state_dict(states["model_state_dict"])
    optimizer.load_state_dict(states["optimizer_state_dict"])
    print(f"Loaded states from {config['training']['ppo']['states_file']}. step: {step}. num_epoch: {num_epoch}")
  else:
    print(f"States file {config['training']['ppo']['states_file']} does not exist. Starting from scratch.")
  
  optimizer.zero_grad()    
  ref_model = deepcopy(model)
  ref_model.eval().cpu()
  
  return {
    "reward_model": reward_model,
    "model": model,
    "ref_model": ref_model,
    "optimizer": optimizer,
    "step": step,
    "num_epoch": num_epoch,
  }

def compute_rewards(model, ref_model, attention_mask, query_response_ids, query_ids, \
                    scores, device, config):
    with torch.no_grad():
        ref_model.to(device)
        logits, values = model(input_ids=query_response_ids,
                               attention_mask=attention_mask) # b, seq, vocab
        ref_logits, _ = ref_model(input_ids=query_response_ids,
                                  attention_mask=attention_mask) # b, seq, vocab
        # (B, seq - 1, vocab)
        log_softmax = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        ref_softmax = torch.nn.functional.log_softmax(ref_logits[:, :-1, :], dim=-1)
        # b, seq - 1
        labels = query_response_ids[:, 1:] 

        # batch, seq - 1
        logp = torch.gather(log_softmax, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) 
        ref_logp = torch.gather(ref_softmax, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        beta = config["training"]["ppo"]["beta"]
        rewards = - beta * (logp - ref_logp) # batch, seq - 1
        
        masks = attention_mask[:, 1:]
        start = query_ids.shape[1] - 1
        ends = start + masks[:, start:].sum(1)
        masks[:, :start] = 0
        for j in range(len(query_response_ids)):
            end = ends[j]
            masks[j, end:] = 0
            rewards[j, end - 1] += scores[j]
        rewards *= masks
        values[:, :-1] *= masks
        ref_model.cpu().eval()
    return {
      "logp":logp,
      "rewards": rewards,
      "values": values[:, :-1],
      "masks": masks,
    }

def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum()

def masked_var(values, mask):
    mean = masked_mean(values, mask)
    centred_values = values - mean
    return masked_mean(centred_values ** 2, mask)

def masked_whiten(values, mask):
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    whitened += mean
    return whitened

def compute_advantage(rewards, values, masks):
    advantages = torch.zeros_like(rewards, device=rewards.device)
    
    next_gae = 0.0
    seq_length = rewards.shape[-1]
    gamma, lam = 1.0, 0.95
    next_values = 0.0
    # delta(t) = r(t) + V(S(t+1)) - V(S(t))
    # A(t) = delta(t) + gamma * lambda * A(t + 1) 
    with torch.no_grad():
        for t in reversed(range(seq_length)):     
            delta = rewards[:, t] + gamma * next_values - values[:, t]
            next_gae = delta + gamma * lam * next_gae
            advantages[:, t] = next_gae
            next_values = values[:, t]
    advantages = masked_whiten(advantages, masks)

    q_vals = advantages + values
    return {
      "advantages": advantages, 
      "q_vals": q_vals,
    }


def compute_loss(old_logprobs, old_values, logprobs, new_vals, masks, advantages, q_vals, config):
    ratio = torch.exp(logprobs - old_logprobs)
    pg_loss1 = - ratio * advantages
    pg_loss2 = - torch.clamp(ratio, 1 - config["training"]["ppo"]["clip_lower_ratio"], 1 + config["training"]["ppo"]["clip_higher_ratio"]) * advantages
    pg_loss = masked_mean(torch.max(pg_loss1, pg_loss2), masks)

    cliprange_value = config["training"]["ppo"]["value_cliprange"] 
    values_clipped = torch.clamp(
        new_vals,
        old_values - cliprange_value,
        old_values + cliprange_value,
    )
    vf_loss1 = (new_vals - q_vals) ** 2
    vf_loss2 = (values_clipped - q_vals) ** 2    
    v_loss = 0.5 * masked_mean(torch.max(vf_loss1, vf_loss2), masks)
    
    loss = pg_loss + config["training"]["ppo"]["value_loss_coeff"] * v_loss

    # avg_ratio = masked_mean(ratio, masks)
    # if avg_ratio > ratio_threshold:
    #     pg_loss = pg_loss * 0.0
    #     v_loss = v_loss * 0.0
    #     loss = loss * 0.0

    return {
      "loss": loss, 
      "v_loss": v_loss,
    }
        
def train(config):
  seed_everything(config["seed"])
  
  rm_tokenizer = get_tokenizer(config["training"]["rm"]["tokenizer_name_or_path"])
  ppo_tokenizer = get_tokenizer(config["training"]["ppo"]["tokenizer_name_or_path"])
  dataloaders = build_dataloader(ppo_tokenizer, config)
  train_dataloader = dataloaders["train_dataloader"]
  val_dataloader = dataloaders["val_dataloader"]

  device = get_device()
  info = get_preloaded_models_and_optimizer(config, DTYPE_MAP[config["dtype"]], device)
  reward_model = info["reward_model"]
  model = info["model"]
  ref_model = info["ref_model"]
  optimizer = info["optimizer"]
  step = info["step"]
  num_epoch = info["num_epoch"]
  generation_kwargs = dict(
    config["training"]["ppo"]["generation_kwargs"],
    eos_token_id=ppo_tokenizer.eos_token_id,
    pad_token_id=ppo_tokenizer.pad_token_id
  )
  
  current_time = datetime.now().strftime(r"%m%d-%H%M")  
  tb_writer = SummaryWriter(log_dir=f"src/logs/{current_time}")
 
  for epoch in range(config["training"]["ppo"]["num_train_epochs"]):
    batch_iterator = tqdm(train_dataloader, desc=f"Epoch {num_epoch+1}", leave=False)
    for batch in batch_iterator:
      gc.collect()
      torch.cuda.empty_cache()
      model.train()       
  
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
      with torch.no_grad():
        reward_model.to(device).eval()
        scores = torch.sigmoid(reward_model(rm_encoded_query_response["input_ids"].to(device), rm_encoded_query_response["attention_mask"].to(device)).logits.squeeze(1))
        scores = 2 * (scores - 0.5)
        reward_model.cpu()
      gc.collect()
      torch.cuda.empty_cache()
      tb_writer.add_scalar("avg rewards", scores.mean().item(), step)


      info = compute_rewards(model, ref_model, attention_mask, query_response_ids, query_ids, \
                      scores, device, config)
      gc.collect()
      torch.cuda.empty_cache()
      
      logp = info["logp"]
      rewards = info["rewards"]
      values = info["values"]
      masks = info["masks"]
      
      info = compute_advantage(rewards, values, masks)
      advantages = info["advantages"]
      q_vals = info["q_vals"]
      
      new_logits, new_values = model(input_ids=query_response_ids,
                               attention_mask=attention_mask) # b, seq, vocab

      # (B, seq - 1, vocab)
      new_log_softmax = torch.nn.functional.log_softmax(new_logits[:, :-1, :], dim=-1)
      # b, seq - 1
      labels = query_response_ids[:, 1:] 

      # batch, seq - 1
      new_logp = torch.gather(new_log_softmax, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)       
      
      info = compute_loss(logp, values, new_logp, new_values[:,:-1], masks, advantages, q_vals, config)
      loss = info["loss"]
      v_loss = info["v_loss"]
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = config["training"]["ppo"]["max_norm"])

      if step % config["training"]["ppo"]["gradient_accumulation_steps"] == 0:
        optimizer.step()
        optimizer.zero_grad()  
        max_memory = torch.cuda.max_memory_allocated()
        tb_writer.add_scalar("max_memory(GB)", max_memory / 1024 ** 3, step)
        torch.cuda.reset_peak_memory_stats()
        
      batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})  
      tb_writer.add_scalar("loss", loss.item(), step)
      tb_writer.add_scalar("v_loss", v_loss.item(), step)
      
      # tb_writer.flush()        
        
      step += 1



    file_name = f'src/ckpt/ppo/ppo_model_epoch{num_epoch:02d}_states.pt'
    torch.save(
            {  
               "step": step,
                "num_epoch": num_epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "model_state_dict": model.state_dict(),
            },
            file_name,
        )     
    num_epoch += 1     
      
      
      
      
      