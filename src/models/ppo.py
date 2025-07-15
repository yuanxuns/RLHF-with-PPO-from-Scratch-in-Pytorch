from torch import nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from copy import deepcopy
from src.models.memory_efficient_adam import MemoryEfficientAdamW
import numpy as np
from pathlib import Path
import sys
import torch


class ValueHead(nn.Module):
    def __init__(self, llm_config, device):
        """
        Initialize a ValueHead.

        Args:
            llm_config: The configuration of the pre-trained language model.
            device: The device to put the value head on.

        The value head is a neural network that takes the hidden state of the
        language model and outputs a scalar value. The architecture of the value
        head is fixed and consists of a LayerNorm, a GELU, a linear layer, a
        GELU, and a final linear layer. The weights of the linear layers are
        initialized using a normal distribution with mean 0 and standard
        deviation 1 / sqrt(max(weight.size()) + 1). The biases of the linear
        layers are initialized to 0.
        """
        super().__init__()
        self.hidden_size = llm_config.hidden_size
        num_labels = 1
        self.value_head = (
            nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, 4 * self.hidden_size),
                nn.GELU(),
                nn.Linear(4 * self.hidden_size, num_labels),
            )
            .to(llm_config.torch_dtype)
            .to(device)
        )

        for layer in self.value_head:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm):
                nn.init.normal_(
                    layer.weight, std=(1.0 / np.sqrt(np.max(layer.weight.size()) + 1))
                )
                nn.init.zeros_(layer.bias)

    def forward(self, hidden_states):
        return self.value_head(hidden_states)


class ModelForCausalLMWithValueHead(nn.Module):
    def __init__(self, model_path, torch_dtype):
        """
        Initialize a ModelForCausalLMWithValueHead.

        Args:
            model_path: The path to the pre-trained language model.
            torch_dtype: The dtype of the model.

        The value head is a neural network that takes the hidden state of the
        language model and outputs a scalar value.
        """
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype
        )
        self.value_head = ValueHead(self.llm.config, self.llm.device)

    def forward(self, input_ids, attention_mask):
        """
        Perform a forward pass through the language model and value head.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Mask to avoid performing attention on padding token indices.

        Returns:
            tuple: A tuple containing:
                - llm_logits (torch.Tensor): Logits from the language model of shape (batch_size, sequence_length, vocab_size).
                - values (torch.Tensor): Scalar values from the value head of shape (batch_size).
        """

        llm_outputs = self.llm.forward(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        llm_logits = llm_outputs.logits
        last_hidden_state = llm_outputs.hidden_states[-1]
        values = self.value_head(last_hidden_state).squeeze(-1)
        return llm_logits, values  # (B, seq_len, vocab_size), (B)

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)


def get_preloaded_models_and_optimizer(config, dtype, device):
    """
    Load pre-trained models and optimizer.

    Args:
        config (Dict[str, Any]): Configuration object.
        dtype (torch.dtype): Data type of the model.
        device (torch.device): Device to place the model.

    Returns:
        A dictionary containing the pre-trained models and optimizer.
    """
    rm_file_path = Path(config["training"]["ppo"]["pretrained_reward_model_path"])
    if not rm_file_path.exists():
        print("Error: Pretrained reward model file is missing.")
        sys.exit(0)

    reward_model = (
        AutoModelForSequenceClassification.from_pretrained(
            config["training"]["ppo"]["pretrained_reward_model_path"],
            torch_dtype=dtype,
            num_labels=1,
        )
        .cpu()
        .eval()
    )

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
    num_epoch = 0
    file_path = Path(config["training"]["ppo"]["states_file"])
    if file_path.exists():
        states = torch.load(config["training"]["ppo"]["states_file"])
        step = states["step"]
        num_epoch = states["num_epoch"]
        model.load_state_dict(states["model_state_dict"])
        optimizer.load_state_dict(states["optimizer_state_dict"])
        print(
            f"Loaded states from {config['training']['ppo']['states_file']}. step: {step}. num_epoch: {num_epoch}"
        )
    else:
        print(
            f"States file {config['training']['ppo']['states_file']} does not exist. Starting from scratch."
        )

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


def compute_rewards(
    model,
    ref_model,
    attention_mask,
    query_response_ids,
    query_ids,
    scores,
    device,
    config,
):
    """
    Compute rewards for PPO.

    Args:
        model (Model): The policy model.
        ref_model (Model): The reference model.
        attention_mask (torch.Tensor): The attention mask of the input sequence.
        query_response_ids (torch.Tensor): The sequence of query and response.
        query_ids (torch.Tensor): The sequence of query.
        scores (torch.Tensor): The rewards for each query.
        device (torch.device): The device to run the computation.
        config (dict): The configuration of the PPO algorithm.

    Returns:
        A dictionary containing the following keys:
        - logp: The log probability of the sequence.
        - rewards: The rewards for each step of the sequence.
        - values: The value of each step of the sequence.
        - masks: The mask of the sequence.
    """
    with torch.no_grad():
        ref_model.to(device)
        logits, values = model(
            input_ids=query_response_ids, attention_mask=attention_mask
        )  # b, seq, vocab
        ref_logits, _ = ref_model(
            input_ids=query_response_ids, attention_mask=attention_mask
        )  # b, seq, vocab
        # (B, seq - 1, vocab)
        log_softmax = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        ref_softmax = torch.nn.functional.log_softmax(ref_logits[:, :-1, :], dim=-1)
        # b, seq - 1
        labels = query_response_ids[:, 1:]

        # batch, seq - 1
        logp = torch.gather(log_softmax, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        ref_logp = torch.gather(
            ref_softmax, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        beta = config["training"]["ppo"]["beta"]
        rewards = -beta * (logp - ref_logp)  # batch, seq - 1

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
        "logp": logp,
        "rewards": rewards,
        "values": values[:, :-1],
        "masks": masks,
    }


def masked_mean(values, mask):
    """
    Compute the mean of values, considering only the masked elements.

    Args:
        values (torch.Tensor): A tensor containing the values to average.
        mask (torch.Tensor): A binary mask tensor with the same shape as `values`,
                             where elements with a value of 1 are included in the mean
                             calculation and elements with a value of 0 are ignored.

    Returns:
        torch.Tensor: The mean of the masked values.
    """

    return (values * mask).sum() / mask.sum()


def masked_var(values, mask):
    """
    Compute the variance of values, considering only the masked elements.

    Args:
        values (torch.Tensor): A tensor containing the values to calculate the variance.
        mask (torch.Tensor): A binary mask tensor with the same shape as `values`,
                             where elements with a value of 1 are included in the variance
                             calculation and elements with a value of 0 are ignored.

    Returns:
        torch.Tensor: The variance of the masked values.
    """
    mean = masked_mean(values, mask)
    centred_values = values - mean
    return masked_mean(centred_values**2, mask)


def masked_whiten(values, mask):
    """
    Whitens the values in a tensor, considering only the elements where mask is 1.

    This function computes the mean and variance of the values, considering only the
    masked elements, and then normalizes the values by subtracting the mean and dividing
    by the square root of the variance. The resulting values are then shifted by the mean
    to preserve the original mean.

    Args:
        values (torch.Tensor): The tensor containing the values to whiten.
        mask (torch.Tensor): A binary mask tensor with the same shape as `values`,
                             where elements with a value of 1 are included in the mean
                             and variance calculation and elements with a value of 0
                             are ignored.

    Returns:
        torch.Tensor: The whitened values.
    """

    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    whitened += mean
    return whitened


def compute_advantage(rewards, values, masks):
    """
    Compute the advantage estimates for a given sequence of rewards and values.

    This function calculates the Generalized Advantage Estimation (GAE) for each time step
    in a sequence, using the provided rewards and value function estimates. The computation
    is performed in reverse order, iterating from the last time step to the first. The
    advantage is calculated using the temporal difference error (delta) at each time step,
    combined with the advantage from the subsequent time step, scaled by the discount and
    GAE lambda factors.

    Args:
        rewards (torch.Tensor): A tensor containing the rewards received at each time step.
        values (torch.Tensor): A tensor containing the value function estimate for each
                               time step.
        masks (torch.Tensor): A binary mask tensor indicating which elements should be
                              included in the advantage computation.

    Returns:
        dict: A dictionary containing the following keys:
            - "advantages": The computed advantage estimates for each time step.
            - "q_vals": The Q-values, which are the sum of advantages and values.
    """

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


def compute_loss(
    old_logprobs, old_values, logprobs, new_vals, masks, advantages, q_vals, config
):
    """
    Computes the PPO loss function from the given inputs.

    Computes the surrogate loss for policy gradient updates and value function
    updates. The surrogate loss is clipped to prevent large updates.

    Args:
        old_logprobs (torch.Tensor): The log probabilities of the actions in the
            original policy.
        old_values (torch.Tensor): The value function estimates of the original
            policy.
        logprobs (torch.Tensor): The log probabilities of the actions in the new
            policy.
        new_vals (torch.Tensor): The value function estimates of the new policy.
        masks (torch.Tensor): A binary mask tensor indicating which elements should
            be included in the loss computation.
        advantages (torch.Tensor): The advantage estimates of the actions in the
            original policy.
        q_vals (torch.Tensor): The Q-values of the actions in the original policy.
        config (dict): A dictionary containing the hyperparameters for the PPO
            algorithm.

    Returns:
        dict: A dictionary containing the following keys:
            - "loss": The total loss of the policy gradient update and value
                function update.
            - "v_loss": The value function loss.
    """
    ratio = torch.exp(logprobs - old_logprobs)
    pg_loss1 = -ratio * advantages
    pg_loss2 = (
        -torch.clamp(
            ratio,
            1 - config["training"]["ppo"]["clip_lower_ratio"],
            1 + config["training"]["ppo"]["clip_higher_ratio"],
        )
        * advantages
    )
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
