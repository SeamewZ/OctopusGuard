import os
import base64
import pandas as pd
from PIL import Image
import torch
from io import BytesIO
from unsloth import FastVisionModel
from transformers import CLIPProcessor, CLIPModel
from OctopusGuard.src.table_tools import get_tx

MODEL_ID = "unsloth/Qwen2.5-VL-7B-Instruct"

qwen_model, qwen_processor = FastVisionModel.from_pretrained(
    model_name=MODEL_ID,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

def llm_chat(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text_prompt], return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        output_ids = qwen_model.generate(**inputs, max_new_tokens=512)
    torch.cuda.empty_cache()

    response = qwen_processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response.strip()

def vision_chat(image_path: str, prompt: str) -> str:
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    text_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text_prompt], images=[image], return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        output_ids = qwen_model.generate(**inputs, max_new_tokens=512)
    torch.cuda.empty_cache()

    response = qwen_processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response.strip()

def truncate_text(text: str, max_chars: int = 10000) -> str:
    return text[:max_chars] + "\n\n[... content truncated ...]" if len(text) > max_chars else text

def analyze_contract_code(path: str, max_chars: int = 10000) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()
    truncated_code = truncate_text(code, max_chars)
    prompt = f"""You are a professional blockchain smart contract auditor.
Below is a smart contract source code:

{truncated_code}

Please identify and list all vulnerabilities in the contract.
If the code is truncated, state that your analysis is based on partial code.
Do not explain anything else.
"""
    return llm_chat(prompt)

def analyze_kline_image(image_path: str) -> str:
    prompt = """You are a blockchain K-line analysis expert.
Based on the token chart image, determine whether it shows a scam.

Please answer strictly in this format:
Scam: Yes or No
Do not explain anything else.
"""
    return vision_chat(image_path, prompt)

def analyze_transaction_early(file_path: str, K: int = 50, max_chars: int = 10000) -> str:
    df = get_tx(file_path, K)
    sample = df.head(K).to_csv(index=False)
    truncated_sample = truncate_text(sample, max_chars)

    prompt = f"""You are a blockchain fraud detection expert.
Below are the earliest {K} transactions of a token:

{truncated_sample}

Based on these transactions, determine whether this token shows scam behavior.
Please answer strictly in this format:
Scam: Yes or No
Do not explain anything else.
"""
    return llm_chat(prompt)

def unified_analysis(contract_path: str, csv_path: str, image_path: str, K: int = 50) -> dict:
    print("🖼️ Analyzing K-line image...")
    kline_result = analyze_kline_image(image_path)

    print("📊 Analyzing transaction data...")
    transaction_result = analyze_transaction_early(csv_path, K)

    print("🔍 Analyzing smart contract...")
    contract_result = analyze_contract_code(contract_path)

    print("🧠 Performing final unified analysis...")
    
    with open(contract_path, "r", encoding='utf-8', errors='ignore') as f:
        contract_code = truncate_text(f.read())

    df = get_tx(csv_path, K)
    transaction_sample = truncate_text(df.head(K).to_csv(index=False))

    final_prompt = f"""You are a professional blockchain security auditor.
You are given:
1. A token price chart image (provided visually).
2. The earliest {K} transactions:\n{transaction_sample}
3. Smart contract source code:\n{contract_code}

Please analyze whether this token is a scam. If any data appears truncated, state that your decision is based on partial input.

Decision format:
Scam: Yes or No
ScamType: a comma-separated list of scam characteristics (if any)
Do not explain anything else.
"""
    final_result = vision_chat(image_path, final_prompt)

    return {
        "kline_log": kline_result,
        "transaction_log": transaction_result,
        "contract_log": contract_result,
        "final_decision": final_result
    }
