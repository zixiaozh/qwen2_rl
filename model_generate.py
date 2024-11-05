import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "updated_model"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the prompt
prompt = """### Instruction:
暴力やいじめが絡む状況を描写した現実的な日本語の会話データを生成してください。  
いじめや暴力的な行動が明確に表現された特定の状況や会話を反映させてください。  
中学生がそのような内容をどのように感じ、捉えるかを考慮してください。  
会話データは20文字から150文字以内に収め、被害の影響を明確かつ正確に伝える内容にしてください。

### Generate:

"""
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# Generate multiple texts
num_texts = 10  # Number of texts to generate
generated_texts = []

for _ in range(num_texts):
    if _ % 100 == 0:
        print(_)
    
    # Prepare the input
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Check model outputs for invalid probabilities
    with torch.no_grad():
        model_output = model(**model_inputs)
        probs = torch.softmax(model_output.logits, dim=-1)
        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            print("Invalid values detected in probability tensor. Skipping generation for this input.")
            continue  # Skip to the next input if invalid values are detected

    # Generate response if probabilities are valid
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # Decode the generated text
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_texts.append(response)


# Save the generated texts to a CSV file
df = pd.DataFrame(generated_texts, columns=["Generated Text"])
df.to_csv("generated_texts_trained.csv", index=False)

print("CSV file with generated texts has been saved.")
