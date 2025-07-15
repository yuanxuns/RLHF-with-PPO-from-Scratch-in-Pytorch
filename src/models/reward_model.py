from transformers import AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


def get_reward_model(
    model_name_or_path: str, device: torch.device, dtype: torch.dtype
) -> AutoModelForSequenceClassification:
    """
    Get the reward model for a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.

    Returns:
        PreTrainedModel: The reward model for the specified model.
    """

    print(f"Building reward model from {model_name_or_path}")
    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, torch_dtype=dtype, num_labels=1
    ).to(device)


def compute_loss(batch, model, device):
    chosen_input_ids = batch["chosen_input_ids"].to(device)
    chosen_attention_mask = batch["chosen_attention_mask"].to(device)
    rejected_input_ids = batch["rejected_input_ids"].to(device)
    rejected_attention_mask = batch["rejected_attention_mask"].to(device)

    # (B, 1)
    r_chosen = model(chosen_input_ids, attention_mask=chosen_attention_mask).logits
    r_rejected = model(
        rejected_input_ids, attention_mask=rejected_attention_mask
    ).logits

    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss
