from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and move it to the device
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
    torch_dtype="auto"
).to(device)

# Get the model's data type
model_dtype = next(model.parameters()).dtype

# Load the processor
processor = AutoProcessor.from_pretrained("huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated")

# Define paths and prompt
video_path = "temp/big-buck-bunny-33s.mp4"
output_path = "temp/big-buck-bunny-33s.txt"
prompt = "Describe this video."

# Prepare the messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": f"file://{video_path}", "max_pixels": 360 * 420, "fps": 1.0},
            {"type": "text", "text": prompt},
        ],
    }
]

# Process inputs
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

# Move inputs to the same device as the model
inputs = {k: v.to(device) for k, v in inputs.items()}

# Store the original default dtype
original_dtype = torch.get_default_dtype()

# Set the default tensor type based on the model's dtype and device
if device == "cuda":
    if model_dtype == torch.float16:
        torch.set_default_tensor_type('torch.cuda.HalfTensor')
    elif model_dtype == torch.float32:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    elif model_dtype == torch.bfloat16:
        torch.set_default_tensor_type('torch.cuda.BFloat16Tensor')
    else:
        raise ValueError(f"Unsupported model dtype: {model_dtype}")
elif device == "cpu":
    if model_dtype == torch.float32:
        torch.set_default_tensor_type('torch.FloatTensor')
    elif model_dtype == torch.float16:
        torch.set_default_tensor_type('torch.HalfTensor')
    elif model_dtype == torch.bfloat16:
        torch.set_default_tensor_type('torch.BFloat16Tensor')
    else:
        raise ValueError(f"Unsupported model dtype: {model_dtype}")

# Generate the output
generated_ids = model.generate(**inputs, max_new_tokens=256)

# Restore the original default dtype
torch.set_default_dtype(original_dtype)
if device == "cpu":
    torch.set_default_tensor_type('torch.FloatTensor')  # Default to float32 for CPU
else:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Default to float32 for CUDA

# Decode and trim the output
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

# Save to file
with open(output_path, "w") as f:
    f.write(output_text[0])

print(f"Description saved to {output_path}")