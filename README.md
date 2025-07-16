# RLHF + PPO: Aligning LLMs with Human Feedback

RLHF is a two-stage process designed to fine-tune language models to align with human preferences. It involves:

### 1. Reward Model Training 

1.1 **Collect preference data**  
   - Human annotators compare pairs (or triples) of completions and determine the preference ranking based on criteria like usefulness, clarity, and tone. In this project, we use imdb dataset https://huggingface.co/datasets/stanfordnlp/imdb.

1.2 **Train the reward model**  
   - A pretrained encoder or LLM is augmented with a scalar regression head.
   - For each pair `(prompt, y_winner, y_loser)`, define a cross-entropy loss:
<p align="center">     
<img width="509" height="44" alt="image" src="https://github.com/user-attachments/assets/4a0fcac3-b75a-41e0-bb0e-eec6e2b81973" />
</p>


### 2. Policy Model Training via PPO 

Once the reward model is trained:

2.1 **Generate rollouts**  
   - Sample prompts from a dataset, generate completions using the current policy model, and score each with the reward model.

2.2 **Add KL penalty**  
   - To prevent the policy model from drifting too far from the reference model, include a per-token KL-divergence penalty in the reward:

<p align="center">     
<img width="493" height="58" alt="image" src="https://github.com/user-attachments/assets/a508536a-7c8c-4ebc-92bb-cfecbbbbacc4" />
</p>


2.3 **Compute advantages and apply PPO**  
   - Use a critic value function netword `V(x)` to estimate value, and compute advantages.
   - Optimize policy using the clipped surrogate objective.
<p align="center">     
<img width="678" height="73" alt="Screenshot 2025-07-15 at 2 58 47 PM" src="https://github.com/user-attachments/assets/31341f79-07f8-4c5c-a37b-2bcdb311aeb1" />
</p>     


2.4 **Update critic (value function)**  
   - Minimize mean-squared TD-error.
    

2.5 **Pseudocode and Implementation Details**  
<p align="center">     
<img width="676" height="427" alt="image" src="https://github.com/user-attachments/assets/b5653d3e-8709-4fe4-96f0-8708592b1ceb" />
</p>

  - Value network is implemented as an additional head on top of the LLM backbone
  
  - Handles different tokenizers of the reward model and the policy/value model
  
  - Use a memory-efficient AdamW optimizer that sets states on CPU
    
  - Training is done on one Nvidia 3060 GPU

______________________


## 3. Pipeline Overview

| Stage       | Input                         | Output                                 |
|-------------|-------------------------------|----------------------------------------|
| Reward      | Preference pairs (prompt, winner, loser) | Trained reward model `R_ϕ`             |
| Policy      | Prompts, `R_ϕ`, and reference policy `π_SFT` | Aligned policy `π_RL` via PPO          |

- **Reward model** stays fixed during policy training.
- **Policy model** is updated using PPO and a KL penalty to balance alignment and distributional faithfulness.
- **Critic** stabilizes training with advantage estimation.


## 4. Reward Model and Policy Model Training

4.1 Update training parameters in
```
src/config/ppo.yaml
```

4.2 Train the reward model via
```
python run_reward_model_trainer.py
```

4.3 Train the policy network via
```
python run_ppo_trainer.py
```

## 5. Training Results
5.1 Reward Model Training Tensorboard
Reward model loss
<img width="963" height="421" alt="rm_loss" src="https://github.com/user-attachments/assets/bc336a47-6b7c-439e-9a64-d33550ad6e3e" />

5.2 Policy and Value Models (PPO) Training Tensorboard

PPO loss
<img width="1064" height="437" alt="training_loss" src="https://github.com/user-attachments/assets/d91b240b-4760-4822-b65c-6c2ad9b2ba72" />
Value network TD error
<img width="1149" height="418" alt="td_error" src="https://github.com/user-attachments/assets/cbd9a8c1-828b-45bc-91cd-f94aeebfd323" />
Generated sequences' average rewards during training
<img width="1157" height="436" alt="avg_rewards" src="https://github.com/user-attachments/assets/410537ef-ad8c-4a00-8581-abe409f14859" />
Average generated sequences' rewards during evaluation
<img width="1160" height="445" alt="eval_avg_rewards" src="https://github.com/user-attachments/assets/3eb2f604-90b0-48e6-b6f3-d8558c14adb1" />

## 6. References

PPO paper: Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O., 2017. Proximal policy optimization algorithms. https://arxiv.org/abs/1707.06347

InstructGPT paper: Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A. and Schulman, J., 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35, pp.27730-27744. https://arxiv.org/abs/2203.02155

Generalized Advantage Estimation paper: Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P., 2015. High-dimensional continuous control using generalized advantage estimation. https://arxiv.org/abs/1506.02438

https://spinningup.openai.com/en/latest/algorithms/ppo.html

https://arxiv.org/pdf/2403.17031

https://arxiv.org/pdf/2203.02155

https://github.com/ash80/RLHF_in_notebooks

https://www.youtube.com/watch?v=11M_kfuPJ5I

https://github.com/hkproj/rlhf-ppo
