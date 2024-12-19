#pip install transformers gradio
#pip install torch torchvision torchaudio
#pip install datasets
#pip install 'accelerate>=0.26.0'
from transformers import GPTJForCausalLM, pipeline, AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch

# Check PyTorch and CUDA availability
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

# Load a pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "EleutherAI/gpt-j-6B"  # Lightweight model for testing: gpt2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTJForCausalLM.from_pretrained(model_name, revision="float16", low_cpu_mem_usage=True)
model.to(device)
# Define a pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(generator("Hello", max_length=50, num_return_sequences=1)) #to check if response


