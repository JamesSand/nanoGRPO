import torch
import torch.nn.functional as F
from torch import Tensor
import contextlib
import time
import wandb
import datetime
from collections import defaultdict
from tqdm import tqdm


class GRPO:
    def __init__(
        self,
        model,
        ref_model,
        dataset=None,
        test_dataset=None,
        tokenizer=None,
        group_size=8,
        micro_group_size=2,
        batch_size=1,
        max_iterations=1000,    
        reward_functions=None,
        log_wandb=False,
        dtype=None,
        lr=5e-6,
        weight_decay=0.0,
        beta=0.0,
        epsilon=0.1
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        
        # Ensure tokenizer has pad_token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.dataset = dataset.shuffle(seed=42)
        self.test_dataset = test_dataset
        
        self.data_loader_iter = iter(self.dataset)
        self.group_size = group_size
        self.micro_group_size = micro_group_size   
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.dtype = dtype if dtype is not None else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.beta = beta
        self.epsilon = epsilon
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,weight_decay=weight_decay)
        assert reward_functions is not None, "Must pass reward_functions"
        self.reward_functions: list = reward_functions

        # self.using_lora = True if self.ref_model is None else False
        
        self.using_lora = True
        if self.using_lora:
            self.ref_model = model

        # self.distributed = False
        self.log_wandb = log_wandb
        if self.log_wandb:
            wandb.init(project="nanoGRPO")

        self.metrics = defaultdict(list)

        self.model.to(self.device).to(dtype)
        self.ref_model.to(self.device).to(dtype)

    def get_per_token_logps(self, model, input_ids) -> Tensor:
        logits = model(input_ids=input_ids).logits
        logits = logits[:, :-1, :]  # Shape: [2, 660, 128256]
        input_ids = input_ids[:, 1:]
        logps = F.log_softmax(logits, dim=-1)
        return torch.gather(logps, -1, input_ids.unsqueeze(-1)).squeeze(-1)

    def compute_loss(self, inputs, old_policy_log_probs, reward, mean_rewards, std_rewards, loss_mask) -> Tensor:
        policy_log_probs = self.get_per_token_logps(self.model, inputs)
        
        with (
            self.ref_model.disable_adapter()
            if self.using_lora  
            else contextlib.nullcontext()
        ):
            ref_policy_log_probs = self.get_per_token_logps(self.ref_model, inputs)


        # advantage calculation
        advantage: Tensor = (reward - mean_rewards) / (std_rewards + 1e-6)
        advantage = advantage.reshape(-1, 1)

        # kl divergence calculation
        log_ratios = ref_policy_log_probs - policy_log_probs
        kld = torch.exp(log_ratios) - log_ratios - 1

        policy_ratio = torch.exp(policy_log_probs-old_policy_log_probs.detach())

        loss1 = policy_ratio*advantage
        loss2 = torch.clamp(policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        loss = -torch.min(loss1, loss2)
        loss = (loss * loss_mask).sum(dim=-1)/ (loss_mask.sum(dim=-1) + 1e-6)
        kld = (kld * loss_mask).sum(dim=-1)/ (loss_mask.sum(dim=-1) + 1e-6)
        loss += kld * self.beta
        if self.log_wandb:
            for _kd in kld:
                self.metrics["kld"].append(_kd.mean().item())
        return loss.mean()

    def sample_batch(self):
        # if self.distributed:
        #     return self.distributed_sample_batch()

        inputs_texts = []
        samples = []
        for _ in range(self.batch_size):
            item = next(self.data_loader_iter)
            samples.append(item)
            prompt = item["prompt"]
            formatted = self.tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs_texts.append(formatted)

        encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        prompt_length = input_ids.shape[1]

        input_ids = torch.repeat_interleave(input_ids, self.group_size, dim=0)
        samples = [sample for _ in range(self.group_size) for sample in samples]

        start_time = time.time()
        max_new_tokens = 512
        outputs = self.model.generate(
            input_ids.to(self.device),
            # min_new_tokens=512,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            # repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
        )
        end_time = time.time()
        print(f"Time for generation: {end_time - start_time} seconds")

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        rewards = self.compute_rewards(samples,decoded_outputs)

        loss_mask = torch.zeros(outputs.shape, dtype=torch.bool)

        gen_tokens = outputs[:, prompt_length:]
        valid_gen_mask = gen_tokens != self.tokenizer.pad_token_id
        loss_mask[:, prompt_length:] = valid_gen_mask

        return outputs, rewards.to(self.dtype).float(), loss_mask[:, 1:]

    def compute_rewards(self, samples, responces) -> Tensor:
        """
        Compute rewards for responses using the reward function.
        
        Args:
            samples: List of sample data
            responces: List of response strings
            
        Returns:
            Tensor: Rewards with shape [batch_size, group_size]
        """
        # Initialize rewards structure: [batch_size][group_size]
        rewards = [[] for _ in range(self.batch_size)]
        
        for idx, (sample, resp) in enumerate(zip(samples, responces)):
            reward = self.reward_functions[0](sample, resp)
            rewards[idx % self.batch_size].append(reward)
        
        rewards = torch.tensor(rewards, dtype=self.dtype).to(self.device)
        
        # Log average reward across groups
        avg_rewards = rewards.mean(dim=-1)
        for r in avg_rewards:
            self.metrics[f"reward"].append(r.item())
        
        # Log prompt lengths
        prompt_lengths = [[] for _ in range(self.batch_size)]
        for idx, sample in enumerate(samples):
            prompt_lengths[idx % self.batch_size].append(len(sample["prompt"]))
        
        for idx, pl in enumerate(prompt_lengths):
            self.metrics[f"prompt_length"].append(sum(pl) / len(pl))
        
        return rewards
    
    def log_metrics(self):
        if self.log_wandb:
            idx = self.metrics["idx"][-1]-1
            metrics = {}
            for k, v in self.metrics.items():
                metrics[f"train/{k}"] = v[idx] if len(v) >= idx else v[-1]
                
            wandb.log(metrics)

    def evaluate(self, num_samples=None, eval_batch_size=None):
        """
        Evaluate the model on test_dataset.
        
        Args:
            num_samples: Number of samples to evaluate. If None, evaluate all test_dataset.
            eval_batch_size: Batch size for evaluation. If None, use larger batch for speed.
        
        Returns:
            dict: Dictionary containing average rewards and other metrics
        """
        if self.test_dataset is None:
            print("Warning: test_dataset is None, skipping evaluation")
            return {}
        
        self.model.eval()
        # Use larger batch size for evaluation (default 8 for faster inference)
        eval_batch_size = eval_batch_size or min(8, len(self.test_dataset))
        
        # Determine number of samples to evaluate
        if num_samples is None:
            test_data = list(self.test_dataset)
        else:
            test_data = list(self.test_dataset.select(range(min(num_samples, len(self.test_dataset)))))
        
        all_rewards = []
        
        print(f"\n{'='*50}")
        print(f"Starting evaluation on {len(test_data)} samples with batch_size={eval_batch_size}...")
        print(f"{'='*50}\n")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), eval_batch_size), desc="Evaluating"):
                batch_samples = test_data[i:i + eval_batch_size]
                inputs_texts = []
                
                for sample in batch_samples:
                    prompt = sample["prompt"]
                    formatted = self.tokenizer.apply_chat_template(
                        prompt, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    inputs_texts.append(formatted)
                
                # Tokenize inputs
                encoded = self.tokenizer(inputs_texts, padding=True, return_tensors="pt")
                input_ids = encoded["input_ids"]
                
                # Generate responses (use greedy decoding for faster and more deterministic evaluation)
                outputs = self.model.generate(
                    input_ids.to(self.device),
                    max_new_tokens=512,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,       # Explicitly use greedy search
                    do_sample=False,   # Disable sampling
                )
                
                # Decode outputs
                decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Compute rewards for this batch
                batch_rewards = []
                for sample, resp in zip(batch_samples, decoded_outputs):
                    reward = self.reward_functions[0](sample, resp)
                    batch_rewards.append(reward)
                
                all_rewards.extend(batch_rewards)
                
                # Clear GPU cache periodically
                if (i // eval_batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
        
        self.model.train()
        
        # Compute statistics
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        max_reward = max(all_rewards) if all_rewards else 0
        min_reward = min(all_rewards) if all_rewards else 0
        
        eval_results = {
            "eval/avg_reward": avg_reward,
            "eval/max_reward": max_reward,
            "eval/min_reward": min_reward,
            "eval/num_samples": len(all_rewards)
        }
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Evaluation Results:")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Max Reward: {max_reward:.4f}")
        print(f"  Min Reward: {min_reward:.4f}")
        print(f"  Num Samples: {len(all_rewards)}")
        print(f"{'='*50}\n")
        
        # Log to wandb if enabled
        if self.log_wandb:
            wandb.log(eval_results)
        
        return eval_results

    def train(self, max_iterations=1000, eval_interval=10):
        
        start_time = time.perf_counter()
            
        for idx in tqdm(range(1, max_iterations + 1)):

            x_batch_inputs, rewards, loss_mask = self.sample_batch()

            batch_inputs = x_batch_inputs.reshape(self.batch_size, self.group_size, *x_batch_inputs.shape[1:])
            loss_mask =       loss_mask.reshape(self.batch_size, self.group_size, *loss_mask.shape[1:])

            pi_old = []
            for _, (b_inputs) in enumerate(batch_inputs):
                
                with torch.no_grad():
                    b_old_policy_log_probs = self.get_per_token_logps(self.model, b_inputs.to(self.device)).cpu()
                    torch.cuda.empty_cache()
                    pi_old.append(b_old_policy_log_probs)

            for _, (b_inputs,b_old_policy_log_probs, b_reward, b_loss_mask) in enumerate(zip(batch_inputs, pi_old, rewards, loss_mask)):
                idx += 1
                reward = b_reward.to(self.device)
                mean_rewards = reward.mean(dim=-1).unsqueeze(-1)
                std_rewards = reward.std(dim=-1).unsqueeze(-1)

                # even grop are too big for vram
                # so we split them into micro groups (its same as micro batching)
                g_inputs                =                b_inputs.reshape(b_inputs.shape[0]//self.micro_group_size,self.micro_group_size, *b_inputs.shape[1:]).cpu()
                g_old_policy_log_probs  =  b_old_policy_log_probs.reshape(b_inputs.shape[0]//self.micro_group_size,self.micro_group_size, *b_old_policy_log_probs.shape[1:]).cpu()
                g_reward =                               b_reward.reshape(b_inputs.shape[0]//self.micro_group_size,self.micro_group_size, *b_reward.shape[1:]).cpu()
                g_loss_mask =                         b_loss_mask.reshape(b_inputs.shape[0]//self.micro_group_size,self.micro_group_size, *b_loss_mask.shape[1:]).cpu()
                group_losses = []
                

                for inputs, old_policy_log_probs, reward, loss_mask in zip(g_inputs, g_old_policy_log_probs, g_reward, g_loss_mask):

                    inputs = inputs.to(self.device)
                    old_policy_log_probs = old_policy_log_probs.to(self.device)
                    reward = reward.to(self.device)
                    loss_mask = loss_mask.to(self.device)

                    loss = self.compute_loss(
                        inputs,
                        old_policy_log_probs,
                        reward,
                        mean_rewards,
                        std_rewards,
                        loss_mask
                    )
                    group_losses.append(loss.item())
                    loss.backward() 

                self.optimizer.step()
                self.optimizer.zero_grad()

                print(f"{idx:04d} loss: {sum(group_losses)/len(group_losses)} reward: {reward.mean()}")
                if self.log_wandb:
                    self.metrics["idx"].append(idx)
                    self.metrics["total_reward"].append(reward.mean().item())
                    self.metrics["loss"].append(sum(group_losses)/len(group_losses))
              
              
            if idx % eval_interval == 0:
                eval_results = self.evaluate(num_samples=40)
                
            print(f"iter {idx}  >>> reward: {rewards.mean()}")
            print(f"Total time: {str(datetime.timedelta(seconds=int(time.perf_counter() - start_time)))}")
            self.log_metrics()