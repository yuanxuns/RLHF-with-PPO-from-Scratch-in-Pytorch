from src.models.reward_model import get_reward_model
from src.tokenizer.tokenizer import get_tokenizer
from src.dataset.rm_dataset import build_dataloader
from src.models.memory_efficient_adam import MemoryEfficientAdamW
from src.models.utils import get_device, seed_everything, DTYPE_MAP
from tqdm import tqdm
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from src.eval.eval import run_rm_eval
from pathlib import Path
import gc
from src.models.reward_model import compute_loss


def train(config):
    """
    Train the reward model using the given configuration.

    This function trains the reward model using the given configuration, saves the
    model's state and optimizer's state to a file at the end of each epoch, and logs
    evaluation metrics to a TensorBoard writer.

    Args:
        config (Dict[str, Any]): Configuration object containing hyperparameters,
            model paths, and other settings.

    Returns:
        None
    """
    seed_everything(config["seed"])

    tokenizer = get_tokenizer(config["training"]["rm"]["tokenizer_name_or_path"])
    dataloader_dict = build_dataloader(tokenizer, config)
    train_dataloader = dataloader_dict["train_dataloader"]
    test_dataloader = dataloader_dict["test_dataloader"]

    device = get_device()
    model = get_reward_model(
        config["training"]["rm"]["model_name_or_path"],
        device=device,
        dtype=DTYPE_MAP[config["dtype"]],
    )
    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=float(config["training"]["rm"]["learning_rate"]),
        weight_decay=config["training"]["rm"]["weight_decay"],
        betas=config["training"]["rm"]["betas"],
        enabled=config["training"]["rm"]["enable_memory_efficient_adamw"],
    )

    current_time = datetime.now().strftime(r"%m%d-%H%M")
    tb_writer = SummaryWriter(log_dir=f"src/logs/{current_time}")

    step = 1
    num_epoch = 1

    file_path = Path(config["training"]["rm"]["states_file"])
    if file_path.exists():
        states = torch.load(config["training"]["rm"]["states_file"])
        step = states["step"]
        num_epoch = states["num_epoch"]
        optimizer.load_state_dict(states["optimizer_state_dict"])
        print(
            f"Loaded states from {config['training']['rm']['states_file']}. step: {step}. num_epoch: {num_epoch}"
        )
    else:
        print(
            f"States file {config['training']['rm']['states_file']} does not exist. Starting from scratch."
        )

    for epoch in range(config["training"]["rm"]["num_train_epochs"]):
        batch_iterator = tqdm(
            train_dataloader, desc=f"Epoch {num_epoch+1}", leave=False
        )

        for batch in batch_iterator:
            gc.collect()
            torch.cuda.empty_cache()

            model.train()

            loss = compute_loss(batch, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config["training"]["rm"]["max_norm"]
            )

            if step % config["training"]["rm"]["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            tb_writer.add_scalar("loss", loss.item(), step)
            # tb_writer.flush()

            if step % config["training"]["rm"]["eval_interval"] == 0:
                run_rm_eval(
                    model,
                    test_dataloader,
                    device,
                    tb_writer,
                    step,
                    config["training"]["rm"]["eval_batch_size"],
                )

            step += 1

        path = f"src/ckpt/rm/reward_model_epoch{num_epoch:02d}"
        model.save_pretrained(path)
        file_name = f"{path}/states.pt"
        torch.save(
            {
                "step": step,
                "num_epoch": num_epoch,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            file_name,
        )
        num_epoch += 1
