import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, create_reference_model, AutoModelForCausalLMWithValueHead

# Set environment variable for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Define custom model class to include a Value Head
class QWenWithValueHead(AutoModelForCausalLMWithValueHead):
    transformers_parent_class = AutoModelForCausalLM  # Ensure this aligns with Qwen model's structure
    lm_head_namings = ["lm_head", "embed_out"]

# Initialize the model and tokenizer
model = QWenWithValueHead.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a reference model for PPO
ref_model = create_reference_model(model)

# Configure PPO
ppo_config = PPOConfig(
    model_name=model_name,
    steps=1000,
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=2,
)

# Initialize PPOTrainer
ppo_trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    config=ppo_config,
)

# Define reward function
def reward_function(text):
    word_count = len(text.split())
    return 1.0 if 20 <= word_count <= 150 else -1.0

# Load data from CSV
csv_path = "generated_texts.csv"
data = pd.read_csv(csv_path)

fixed_prompt = """### Instruction:
暴力やいじめが絡む状況を描写した現実的な日本語の会話データを生成してください。  
いじめや暴力的な行動が明確に表現された特定の状況や会話を反映させてください。  
中学生がそのような内容をどのように感じ、捉えるかを考慮してください。  
会話データは20文字から150文字以内に収め、被害の影響を明確かつ正確に伝える内容にしてください。

### Generate:
{}"""

# Training loop using data from CSV
batch_prompts, batch_responses, batch_rewards = [], [], []

for step, row in enumerate(data.itertuples(), start=1):
    # Set the prompt text and response from the CSV column
    prompt_text = fixed_prompt.format("")
    response_text = row.Generated_Text

    # Compute reward for the response
    reward = reward_function(response_text)

    # Accumulate batches
    batch_prompts.append(prompt_text)
    batch_responses.append(response_text)
    batch_rewards.append(reward)

    # Execute PPO training step if batch size is met
    if len(batch_prompts) == ppo_config.batch_size:
        try:
            # Tokenize with padding to the maximum length within this batch and send to device
            # Initialize device
            device = next(model.parameters()).device

            # Tokenize and convert to lists of tensors
            batch_prompt_tensors = [
                tensor for tensor in tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).input_ids.to(device)
            ]

            batch_response_tensors = [
                tensor for tensor in tokenizer(
                    batch_responses,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).input_ids.to(device)
            ]

            batch_reward_tensors = [torch.tensor(reward, device=device) for reward in batch_rewards]

            # Execute PPO step
            ppo_trainer.step(batch_prompt_tensors, batch_response_tensors, batch_reward_tensors)

            
        except RuntimeError as e:
            print(f"PPO step error at step {step}: {e}")
            break

        # Clear batches
        batch_prompts, batch_responses, batch_rewards = [], [], []

    if step % 100 == 0:
        print(f"Step {step}: Reward - {reward}")

# Save the updated model
model.save_pretrained("updated_model")
tokenizer.save_pretrained("updated_model")
