
import os
import sys
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Disable CUDA and telemetry checks
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

MODEL_ID = os.environ.get("MODEL_ID", "qurrat99/qwen2.5-0.5b-dpo-lab10")
print(f"Initializing app.py - target model: {MODEL_ID}")

tokenizer = None
model = None
load_error = None

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model on CPU (FP32)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).eval()
    print("Model successfully loaded!")
except Exception as e:
    load_error = str(e)
    print(f"CRITICAL ERROR during initialization: {load_error}", file=sys.stderr)

def generate(prompt):
    if load_error:
        return f"Model failed to load. Error details: {load_error}"
    if not model or not tokenizer:
        return "Model or tokenizer is not initialized."
        
    prompt = (prompt or "").strip()[:1000]
    if not prompt:
        return ""
    
    text = (
        "Write a short, factual response in natural language.\n"
        f"Topic: {prompt}\n"
        "Response:\n"
    )
    try:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        return f"Generation Error: {str(e)}"

demo = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Prompt", lines=3),
    outputs=gr.Textbox(label="Response"),
    title="DPO-Aligned Qwen2.5-0.5B (CPU)",
    api_name="generate",
)
demo.queue(max_size=8).launch()
