from transformers import AutoModelForSequenceClassification
import torch


def get_reward_model(model_name_or_path: str, device: torch.device, dtype:torch.dtype) -> AutoModelForSequenceClassification:
    """
    Get the reward model for a given model name or path.
    
    Args:
        model_name_or_path (str): The name or path of the model.
        
    Returns:
        PreTrainedModel: The reward model for the specified model.
    """

    print(f"Building reward model from {model_name_or_path}")
    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype, 
        num_labels=1     
    ).to(device)
