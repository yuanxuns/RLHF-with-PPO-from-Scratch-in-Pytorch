
from transformers import AutoTokenizer


def get_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    """
    Get the tokenizer for a given model name or path.
    
    Args:
        model_name_or_path (str): The name or path of the model.
        
    Returns:
        PreTrainedTokenizer: The tokenizer for the specified model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

