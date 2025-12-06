import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"  
LORA_REPO = "pontusluthi/iris" 
device = "cpu" 

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model on CPU (this can take a while)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map={"": device},
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_REPO)
model.to(device)
model.eval()

print("Model loaded successfully!")

def generate_response(prompt):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and return
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create Gradio interface
with gr.Blocks(title="LoRA Model Inference") as demo:
    gr.Markdown("#LoRA Model Inference")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=5,
            )
            
            submit_btn = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Response",
                lines=10,
                interactive=False,
            )
    
    # Event handlers
    submit_btn.click(
        fn=generate_response,
        inputs=[prompt_input],
        outputs=output_text,
    )
    
    
    # Example prompts
    gr.Examples(
        examples=[
            ["Tell me a story about a brave knight."],
            ["Explain quantum computing in simple terms."],
            ["Write a haiku about nature."],
        ],
        inputs=prompt_input,
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)

