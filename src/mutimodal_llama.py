import os
from PIL import Image
import torch
import faiss
import pickle
import numpy as np
from transformers import (
    AutoProcessor,
    CLIPProcessor,
    CLIPModel
)
from unsloth import FastVisionModel
from OctopusGuard.src.table_tools import extract_transaction_features

MODEL_ID = "OctopusGuard/src/llama32vision_finetune_checkpoint/checkpoint-2130"
CLIP_PATH = "openai/clip-vit-base-patch32"

qwen_model, qwen_processor = FastVisionModel.from_pretrained(
    model_name = MODEL_ID,
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

clip_model = CLIPModel.from_pretrained(CLIP_PATH).to("cuda")
clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH)


VECTOR_DB_PATH = "OctopusGuard/data_processing/klines_rag_db/vector_db/klines_faiss.index"
META_PATH = "OctopusGuard/data_processing/klines_rag_db/vector_db/metadata.pkl"

index = faiss.read_index(VECTOR_DB_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)


def llm_chat(text: str) -> str:
    messages = [{"role": "user", "content": text}]
    prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[prompt], return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        output_ids = qwen_model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.8)
    torch.cuda.empty_cache()

    response = qwen_processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response.strip()

def number_contract_lines(code: str) -> str:
    return "\n".join([f"{i+1} {line}" for i, line in enumerate(code.strip().splitlines())])


def analyze_contract_code(path: str) -> tuple[str, str]:
    with open(path, "r") as f:
        code = f.read()
    numbered_code = number_contract_lines(code)
    ctx = numbered_code + "\n\n"

    questions = [
        """Help me identify the critical program points and write the assertion invariants in the following format, one per line:\n\nline_number+ assert(exp1 op exp2);\n\nOnly output the list of assertions. Do not include code block markers, explanations, or any extra text.""",
        """Given the following inferred critical invariants:\n\n{}\n\nRank them in descending order of importance based on security criticality, impact, or control relevance.\n\nRespond with the sorted list in the same format, one per line, exactly as input, and do not include any explanation, code block markers, or extra text.""",
        """What are the vulnerabilities in the contract?\n\nDo not include commas, code blocks, or any explanation."""
    ]

    answers = []
    for i, q_template in enumerate(questions):
        if '{}' in q_template:
            previous_answer = answers[-1][1].strip() if answers else ""
            q = q_template.format(previous_answer)
        else:
            q = q_template
        
        a = llm_chat(ctx + q)
        answers.append((q, a))

    q_and_a_text = "\n\n".join([f"Q: {q.strip()}\n\nA: {a.strip()}" for q, a in answers])
    full_log = "\n📝 Contract Analysis Dialogue Log:\n" + q_and_a_text

    clean_answers = "\n".join([a.strip() for q, a in answers])

    return full_log, clean_answers


def get_image_embedding(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy()

def search_similar_images(query_vector: np.ndarray, top_k: int = 3, threshold: float = 0.9):
    scores, indices = index.search(query_vector, top_k)
    results = []
    for i, score in zip(indices[0], scores[0]):
        similarity = 1 - (score / 2) 
        if similarity >= threshold and i < len(metadata):
            entry = metadata[i]
            results.append({
                "image_path": entry["image_path"],
                "label": entry["label"],
                "pump_and_dump": entry["pump_and_dump"]
            })
    return results

def build_kline_prompt(similar_data) -> str:
    prompt = (
    "You are a blockchain image-based financial analysis expert.\n"
    "Analyze the provided token price chart. Use the following labels from historically similar charts as additional context to inform your decision. "
    "Then determine whether it is a scam.\n\n"
    )
    if similar_data:
        prompt += "Reference from similar charts::\n"
        for i, entry in enumerate(similar_data):
            pump_label = "pump_and_dump" if entry["pump_and_dump"] else "not pump_and_dump"
            prompt += f"Similar Chart {i+1}: PumpAndDump='{pump_label}', ScamLabel='{entry['label']}', \n"
        prompt += "first determine whether the input token is involved in pump_and_dump schemes.\nThen determine whether it is a scam."
    else:
        prompt += "No similar charts found. Model will analyze based on the input image only.\n"
    
    prompt += (
        "\nPlease only answer in the following strict format:\n"
        "PumpAndDump: Yes or No\n"
        "Scam: Yes or No\n"
    )
    return prompt

def analyze_kline_image(image_path: str) -> tuple[str, str]:
    query_vector = get_image_embedding(image_path)
    similar = search_similar_images(query_vector)
    prompt = build_kline_prompt(similar)

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    
    print(f"\n🔹 Q (Image + Text): {prompt.strip()}")
    text_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text_prompt], images=[image], return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=256)
    torch.cuda.empty_cache()

    clean_answer = qwen_processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    print(f"🟢 A: {clean_answer}")

    full_log = f"Q: {prompt.strip()}\n\nA: {clean_answer}"
    
    return full_log, clean_answer


def analyze_transaction_csv(csv_path: str) -> tuple[str, str]:
    feature_explanation_dict = {
        "creator_address": "creator address", "creator_hold_ratio": "The proportion of total tokens held by the contract creator.", "top10_hold_ratio": "The proportion of total tokens held by the top 10 holders.", "top5_hold_ratio": "The proportion of total tokens held by the top 5 holders.", "blackhole_transfer": "Whether liquidity pool tokens have been sent to burn or lock addresses.", "add_liquidity_called": "Whether liquidity has been added to decentralized exchanges.", "method_distribution": "The frequency distribution of different method calls in contract interactions.", "lifetime_seconds": "The time elapsed since the token contract was deployed.", "receiver_entropy": "The distribution entropy of receiver addresses after filtering wash trading and airdrops.", "airdrop_detected": "Whether the token was distributed through airdrops.", "airdrop_transactions_count": "The total number of airdrop transactions.", "wash_trading_detected": "Indicates whether artificial trading activity (e.g., self-trading to inflate volume) is present.", "wash_trading_pairs_count": "Number of token pairs involved in wash trading.", "wash_trading_trascation_nums": "Total number of wash trading transactions.", "buy_user_count": "The number of unique addresses that purchased the token.", "sell_user_count": "The number of unique addresses that sold the token.", "sell_user_ratio": "The ratio of sellers to all users.", "buy_user_ratio": "The ratio of buyers to all users.", "low_slippage_user_count": "The number of users who sold with unusually low slippage.", "low_slippage_users_ratio": "The ratio of users selling with very low slippage.", "top_addresses_tx_count": "Transaction volume of top token holders.", "top_addresses_tx_ratio": "The proportion of total transactions made by top token holders.", "total_transactions": "The total number of transactions.", "total_supply": "The total number of tokens minted by the contract.", "unique_addresses": "The number of unique addresses interacting with the token."
    }

    features = extract_transaction_features(csv_path)
    feature_str = ", ".join([f"{k}: {v}" for k, v in features.items()])
    explanations = "\n".join([f"- {k}: {feature_explanation_dict[k]}" for k in features.keys()])

    prompt = f"""
You are a professional blockchain security analysis assistant.

Below are the extracted transaction features of a token:
{feature_str}

=== Feature Explanation ===
{explanations}

Based on the above features and explanations, analyze the risk:

Please analyze and only answer in the following strict format:

Scam: Yes or No
ScamType: type_of_scam (e.g. honeypot, others)
KeyFeaturesUsed: list of key features and their values

Do not explain anything else.
"""
    
    print(f"\n🔹 Q: {prompt.strip()}")
    clean_answer = llm_chat(prompt)
    print(f"🟢 A: {clean_answer}")

    full_log = f"Q: {prompt.strip()}\n\nA: {clean_answer}"
    
    return full_log, clean_answer

def unified_analysis(contract_path: str, csv_path: str, image_path: str) -> str:
    output_log = ""

    output_log += "\n🖼️ Analyzing token price chart image...\n"
    image_log, image_answer = analyze_kline_image(image_path)
    output_log += f"{image_log}\n"

    output_log += "\n📊 Analyzing transaction data...\n"
    tx_log, tx_answer = analyze_transaction_csv(csv_path)
    output_log += f"{tx_log}\n"

    output_log += "\n🔍 Analyzing smart contract code...\n"
    contract_log, contract_answer = analyze_contract_code(contract_path)
    output_log += f"{contract_log}\n"

    final_summary_prompt = f"""
You are an experienced blockchain security auditor.

Here are the multimodal analysis results for a token:

[Chart Image Analysis]:
{image_answer}

[Transaction Feature Analysis]:
{tx_answer}

[Contract Analysis]:
{contract_answer}

Please synthesize the above results carefully. Use the following decision logic:

- If only one or two modalities indicates scam, output Scam: No.
- If all three modalities indicate scam and agree on the scam type, output Scam: Yes and the scam type.
- If modalities conflict or are inconclusive, output Scam: No.
- Give priority to Chart Image Analysis if strong indicators are present.

Only provide the final answer in the following strict format:

Scam: Yes or No
ScamType: type_of_scam (e.g. honeypot, rugpull, others)

Do not provide any explanation.
"""
    print("\n🧠 Generating Final Assessment with the following prompt:")
    print("--------------------")
    print(final_summary_prompt)
    print("--------------------")
    
    final_decision = llm_chat(final_summary_prompt)
    output_log += "\n🧠 Final Assessment:\n"
    output_log += f"{final_decision}\n"

    return output_log