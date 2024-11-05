import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict
from trl import AutoModelForCausalLMWithValueHead

# Define the custom model class
class QWenWithValueHead(AutoModelForCausalLMWithValueHead):
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]

# Load the updated model and tokenizer using the custom class
model = QWenWithValueHead.from_pretrained("updated_model")
tokenizer = AutoTokenizer.from_pretrained("updated_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.pretrained_model.to(device)  # Explicitly move the pretrained_model
model.eval()

# Sample prompt to tokenize and pass through the model
prompt = """
Hello, how are you today?  
"""

# Tokenize the input
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Display tokens for debugging
print("Tokenized Input Tokens:", tokenizer.convert_ids_to_tokens(input_ids[0]))

# Dictionary to store the outputs of each layer
layer_outputs = OrderedDict()

# Hook function to capture layer outputs
def hook_fn(module, input, output):
    # Handle output if it's a tuple (e.g., when attention weights are returned)
    if isinstance(output, tuple):
        output = output[0]  # Use the first element (hidden states)
    layer_outputs[module] = output

# Register hooks on each layer
hooks = []
for name, layer in model.named_modules():
    # Register hooks only on relevant layers (e.g., transformer layers)
    if "layer" in name or "attention" in name or "mlp" in name:
        hooks.append(layer.register_forward_hook(hook_fn))

# Check for NaNs in the model's initial weights
for name, param in model.named_parameters():
    if param.requires_grad and ("v_head" in name or param.isnan().any()):
        print(f"Reinitializing weights in layer: {name}")
        torch.nn.init.normal_(param, mean=0.0, std=0.02)

# Perform a forward pass through the model to capture outputs at each layer
with torch.no_grad():
    _ = model(input_ids)

# Remove hooks
for hook in hooks:
    hook.remove()

# Display tokenized inputs and intermediate layer outputs
print("Tokenized Input IDs:", input_ids)
# print("Intermediate Layer Outputs:")
# for layer, output in layer_outputs.items():
#     if torch.isnan(output).any() or torch.isinf(output).any():
#         print(f"{layer}: Output contains NaN or Inf values. Attempting reinitialization...")
        
#         # Reinitialize specific layers to address NaN/Inf issues
#         if hasattr(layer, 'reset_parameters'):
#             layer.reset_parameters()
#             print(f"Reset parameters for layer: {layer}")

#     else:
#         print(f"{layer}: Output shape: {output.shape}")
        
#         # Dynamically adjust slicing based on tensor dimensions
#         if output.dim() == 3:
#             print(output[0, :5, :5])  # For 3D tensors
#         elif output.dim() == 2:
#             print(output[0, :5])      # For 2D tensors
#         elif output.dim() == 1:
#             print(output[:5])         # For 1D tensors


# Generate text using sampling
with torch.no_grad():
    generated_ids = model.generate(input_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated Text:", generated_text)
