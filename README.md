# RLHF + PPO: Aligning LLMs with Human Feedback

RLHF is a two-stage process designed to fine-tune language models to align with human preferences. It involves:

### 1. Reward Model Training 🎯

1. **Collect preference data**  
   - Human annotators compare pairs (or triples) of completions and determine the preference ranking based on criteria like usefulness, clarity, and tone. In this project, we use imdb dataset https://huggingface.co/datasets/stanfordnlp/imdb.

2. **Train the reward model**  
   - A pretrained encoder or LLM is augmented with a scalar regression head.
   - For each pair `(prompt, y_winner, y_loser)`, define a cross-entropy loss:
<p align="center">     
<img width="509" height="44" alt="image" src="https://github.com/user-attachments/assets/4a0fcac3-b75a-41e0-bb0e-eec6e2b81973" />
</p>


### 2. Policy Model Training via PPO 

Once the reward model is trained:

1. **Generate rollouts**  
   - Sample prompts from a dataset, generate completions using the current policy model, and score each with the reward model.

2. **Add KL penalty**  
   - To prevent the policy model from drifting too far from the reference model, include a per-token KL-divergence penalty in the reward:

<p align="center">     
<img width="493" height="58" alt="image" src="https://github.com/user-attachments/assets/a508536a-7c8c-4ebc-92bb-cfecbbbbacc4" />
</p>


3. **Compute advantages and apply PPO**  
   - Use a critic value function netword `V(x)` to estimate value, and compute advantages.
   - Optimize policy using the clipped surrogate objective.
<p align="center">     
<img width="678" height="73" alt="Screenshot 2025-07-15 at 2 58 47 PM" src="https://github.com/user-attachments/assets/31341f79-07f8-4c5c-a37b-2bcdb311aeb1" />
</p>     


4. **Update critic (value function)**  
   - Minimize mean-squared TD-error.
    

5. **Pseudocode**  
<p align="center">     
<img width="676" height="427" alt="image" src="https://github.com/user-attachments/assets/b5653d3e-8709-4fe4-96f0-8708592b1ceb" />
</p>


## Pipeline Overview

| Stage       | Input                         | Output                                 |
|-------------|-------------------------------|----------------------------------------|
| Reward      | Preference pairs (prompt, winner, loser) | Trained reward model `R_ϕ`             |
| Policy      | Prompts, `R_ϕ`, and reference policy `π_SFT` | Aligned policy `π_RL` via PPO          |

- **Reward model** stays fixed during policy training.
- **Policy model** is updated using PPO and a KL penalty to balance alignment and distributional faithfulness.
- **Critic** stabilizes training with advantage estimation.



## 📚 Optional Variations & Notes

- **Reward whitening/normalization**, handling EOS tokens, and padding are critical for stability :contentReference[oaicite:8]{index=8}.
- **PPO-max** introduces strengthened policy constraints to improve stability in long training :contentReference[oaicite:9]{index=9}.
- Alternatives like **Direct Preference Optimization (DPO)** and **Constitutional AI** offer bypasses to explicit reward modeling :contentReference[oaicite:10]{index=10}.



## References

https://github.com/ash80/RLHF_in_notebooks

https://www.youtube.com/watch?v=11M_kfuPJ5I

https://github.com/hkproj/rlhf-ppo

PPO paper: Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O., 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. - https://arxiv.org/abs/1707.06347

InstructGPT paper: Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A. and Schulman, J., 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35, pp.27730-27744. - https://arxiv.org/abs/2203.02155

Generalized Advantage Estimation paper: Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P., 2015. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438. - https://arxiv.org/abs/1506.02438

https://spinningup.openai.com/en/latest/algorithms/ppo.html

https://arxiv.org/pdf/2403.17031

https://arxiv.org/pdf/2203.02155
