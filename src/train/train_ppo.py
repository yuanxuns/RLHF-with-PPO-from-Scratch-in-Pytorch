from src.models.utils import get_device, seed_everything, DTYPE_MAP
from src.tokenizer.tokenizer import get_tokenizer

import torch

from tqdm import tqdm
import gc
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from src.eval.eval import run_ppo_eval

from src.dataset.ppo_dataset import build_dataloader
from src.models.ppo import get_preloaded_models_and_optimizer
from src.models.ppo import compute_loss, compute_advantage, compute_rewards


def train(config):
    """
    Train the PPO model using the given configuration.

    This function trains the PPO model using the given configuration and saves the
    model's state and optimizer's state to a file at the end of each epoch.

    Args:
        config (Dict[str, Any]): Configuration object containing hyperparameters,
            model paths, and other settings.

    Returns:
        None
    """
    seed_everything(config["seed"])

    rm_tokenizer = get_tokenizer(config["training"]["rm"]["tokenizer_name_or_path"])
    ppo_tokenizer = get_tokenizer(config["training"]["ppo"]["tokenizer_name_or_path"])
    dataloaders = build_dataloader(ppo_tokenizer, config)
    train_dataloader = dataloaders["train_dataloader"]
    val_dataloader = dataloaders["val_dataloader"]

    device = get_device()
    info = get_preloaded_models_and_optimizer(
        config, DTYPE_MAP[config["dtype"]], device
    )

    reward_model, model, ref_model, optimizer, step, num_epoch = (
        info[k]
        for k in [
            "reward_model",
            "model",
            "ref_model",
            "optimizer",
            "step",
            "num_epoch",
        ]
    )

    generation_kwargs = dict(
        config["training"]["ppo"]["generation_kwargs"],
        eos_token_id=ppo_tokenizer.eos_token_id,
        pad_token_id=ppo_tokenizer.pad_token_id,
    )

    current_time = datetime.now().strftime(r"%m%d-%H%M")
    tb_writer = SummaryWriter(log_dir=f"src/logs/{current_time}")

    for epoch in range(config["training"]["ppo"]["num_train_epochs"]):
        num_epoch += 1
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {num_epoch}", leave=False)
        for batch in batch_iterator:
            gc.collect()
            torch.cuda.empty_cache()
            model.train()

            query_ids = torch.nn.utils.rnn.pad_sequence(
                batch["input_ids"],
                batch_first=True,
                padding_value=ppo_tokenizer.pad_token_id,
            ).to(device)

            query_attention_masks = torch.nn.utils.rnn.pad_sequence(
                batch["attention_mask"],
                batch_first=True,
                padding_value=ppo_tokenizer.pad_token_id,
            ).to(device)

            with torch.no_grad():
                query_response_ids = model.generate(
                    input_ids=query_ids,
                    attention_mask=query_attention_masks,
                    **generation_kwargs,
                )

            # response_ids = query_response_ids[:, query_ids.shape[1]:]
            attention_mask = query_response_ids.not_equal(
                ppo_tokenizer.pad_token_id
            ).long()
            query_response_texts = ppo_tokenizer.batch_decode(
                query_response_ids, skip_special_tokens=True
            )
            rm_encoded_query_response = rm_tokenizer(
                query_response_texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            with torch.no_grad():
                reward_model.to(device).eval()
                scores = torch.sigmoid(
                    reward_model(
                        rm_encoded_query_response["input_ids"].to(device),
                        rm_encoded_query_response["attention_mask"].to(device),
                    ).logits.squeeze(1)
                )
                scores = 2 * (scores - 0.5)
                reward_model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            tb_writer.add_scalar("avg rewards", scores.mean().item(), step)

            info = compute_rewards(
                model,
                ref_model,
                attention_mask,
                query_response_ids,
                query_ids,
                scores,
                device,
                config,
            )
            logp, rewards, values, masks = (
                info[k] for k in ["logp", "rewards", "values", "masks"]
            )
            gc.collect()
            torch.cuda.empty_cache()

            info = compute_advantage(rewards, values, masks)
            advantages, q_vals = info["advantages"], info["q_vals"]

            new_logits, new_values = model(
                input_ids=query_response_ids, attention_mask=attention_mask
            )  # b, seq, vocab

            # (B, seq - 1, vocab)
            new_log_softmax = torch.nn.functional.log_softmax(
                new_logits[:, :-1, :], dim=-1
            )
            # b, seq - 1
            labels = query_response_ids[:, 1:]

            # batch, seq - 1
            new_logp = torch.gather(
                new_log_softmax, dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)

            info = compute_loss(
                logp,
                values,
                new_logp,
                new_values[:, :-1],
                masks,
                advantages,
                q_vals,
                config,
            )
            loss = info["loss"]
            v_loss = info["v_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config["training"]["ppo"]["max_norm"]
            )

            if step % config["training"]["ppo"]["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                max_memory = torch.cuda.max_memory_allocated()
                tb_writer.add_scalar("max_memory(GB)", max_memory / 1024**3, step)
                torch.cuda.reset_peak_memory_stats()

            batch_iterator.set_postfix(
                {
                    "loss": f"{loss.item():.3f}",
                    "avg_rewards": f"{scores.mean().item():.3f}",
                }
            )
            tb_writer.add_scalar("loss", loss.item(), step)
            tb_writer.add_scalar("v_loss", v_loss.item(), step)

            # Evaluate the model periodically.
            if step % config["training"]["ppo"]["eval_interval"] == 0:
                run_ppo_eval(
                    model,
                    reward_model,
                    val_dataloader,
                    ppo_tokenizer,
                    rm_tokenizer,
                    generation_kwargs,
                    tb_writer,
                    device,
                    step,
                )

            # Updates the reference model periodically.
            if step % config["training"]["ppo"]["ref_model_update_interval"] == 0:
                ref_model.load_state_dict(model.state_dict())
                ref_model.eval().cpu()
                print(f"Updated reference model at step {step}")

            # tb_writer.flush()
            step += 1

        file_name = (
            f"src/ckpt/ppo/ppo_model_epoch{num_epoch:02d}_states_{current_time}.pt"
        )
        torch.save(
            {
                "step": step,
                "num_epoch": num_epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "model_state_dict": model.state_dict(),
                "config": config,
            },
            file_name,
        )
