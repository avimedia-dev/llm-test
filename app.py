#pip install transformers gradio
#pip install torch torchvision torchaudio
#pip install datasets
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch

# Check PyTorch and CUDA availability
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

# Load a pre-trained model
model_name = "EleutherAI/gpt-j-6B"  # Lightweight model for testing: gpt2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define a pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(generator("Hello", max_length=50, num_return_sequences=1)) #to check if response

# Conversation function
def conversation(prompt, history=None):
    """
    Handles multi-turn conversations by maintaining context.

    Args:
        prompt (str): User's input.
        history (list): List of previous exchanges.

    Returns:
        tuple: (response, updated history)
    """
    if history is None:
        history = []

    # Add user input to the conversation history
    history.append(f"User: {prompt}")
    input_text = "\n".join(history)

    # Generate the response
    results = generator(input_text, max_length=200, num_return_sequences=1)
    response = results[0]["generated_text"]

    # Extract only the latest assistant's response
    assistant_reply = response.split("\n")[-1]
    history.append(f"Assistant: {assistant_reply.strip()}")

    return assistant_reply.strip(), history

# Define Gradio interface
def gradio_conversation(prompt, history):
    response, updated_history = conversation(prompt, history)
    return response, updated_history

interface = gr.Interface(
    fn=gradio_conversation,
    inputs=[
        gr.Textbox(label="Enter Your Message", placeholder="Type something..."),
        gr.State()  # To keep the conversation history
    ],
    outputs=[
        gr.Textbox(label="Assistant's Response"),
        gr.State()  # To update the conversation history
    ],
    title="Conversational AI",
    description="An example of a conversational chatbot with history using Gradio and a small LLM.",
)

# Launch the interface
interface.launch()