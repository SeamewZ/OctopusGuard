import os
from PIL import Image
import torch
import numpy as np
from transformers import (
    AutoProcessor,
)
from unsloth import FastVisionModel
from OctopusGuard.src.table_tools import extract_transaction_features

class AblationCodeTx: 
    def __init__(self, model_id: str):
        print(f"Initializing Ablation (Code + Tx) with model: {model_id}")
        
        self.qwen_model, self.qwen_processor = FastVisionModel.from_pretrained(
            model_name=model_id,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        
        print("Ablation (Code + Tx) initialized successfully.")

    def llm_chat(self, text: str) -> str:
        messages = [{"role": "user", "content": text}]
        prompt = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.qwen_processor(text=[prompt], return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            output_ids = self.qwen_model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.8)
        torch.cuda.empty_cache()

        response = self.qwen_processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return response.strip()

    def number_contract_lines(self, code: str) -> str:
        return "\n".join([f"{i+1} {line}" for i, line in enumerate(code.strip().splitlines())])

    def analyze_contract_code(self, path: str) -> tuple[str, str]:
        with open(path, "r") as f:
            code = f.read()
        numbered_code = self.number_contract_lines(code)
        ctx = numbered_code + "\n\n"

        vulnerability_question = """What are the vulnerabilities in the contract?\n\nDo not include commas, code blocks, or any explanation."""
        
        print("\n🔹 Q (Contract Code): Analyzing for vulnerabilities...")
        full_prompt = ctx + vulnerability_question
        clean_answer = self.llm_chat(full_prompt)
        print(f"🟢 A: {clean_answer}")
        
        full_log = f"\n📝 Contract Analysis Dialogue Log:\nQ: {vulnerability_question.strip()}\n\nA: {clean_answer}"
        
        return full_log, clean_answer
    
    def analyze_transaction_csv(self, csv_path: str) -> tuple[str, str]:
        feature_explanation_dict = {
            "creator_address": "creator address", "creator_hold_ratio": "The proportion of total tokens held by the contract creator.", "top10_hold_ratio": "The proportion of total tokens held by the top 10 holders.", "top5_hold_ratio": "The proportion of total tokens held by the top 5 holders.", "blackhole_transfer": "Whether liquidity pool tokens have been sent to burn or lock addresses.", "add_liquidity_called": "Whether liquidity has been added to decentralized exchanges.", "method_distribution": "The frequency distribution of different method calls in contract interactions.", "lifetime_seconds": "The time elapsed since the token contract was deployed.", "receiver_entropy": "The distribution entropy of receiver addresses after filtering wash trading and airdrops.", "buy_user_count": "The number of unique addresses that purchased the token.", "sell_user_count": "The number of unique addresses that sold the token.", "sell_user_ratio": "The ratio of sellers to all users.", "buy_user_ratio": "The ratio of buyers to all users.", "low_slippage_user_count": "The number of users who sold with unusually low slippage.", "low_slippage_users_ratio": "The ratio of users selling with very low slippage.", "top_addresses_tx_count": "Transaction volume of top token holders.", "top_addresses_tx_ratio": "The proportion of total transactions made by top token holders.", "total_transactions": "The total number of transactions.", "total_supply": "The total number of tokens minted by the contract.", "unique_addresses": "The number of unique addresses interacting with the token."
        }
        features = extract_transaction_features(csv_path)
        feature_str = ", ".join([f"{k}: {v}" for k, v in features.items()])
        explanations = "\n".join([f"- {k}: {feature_explanation_dict[k]}" for k in features.keys()])
        prompt = f"You are a professional blockchain security analysis assistant.\n\nBelow are the extracted transaction features of a token:\n{feature_str}\n\n=== Feature Explanation ===\n{explanations}\n\nBased on the above features and explanations, analyze the risk:\n\nPlease analyze and only answer in the following strict format:\n\nScam: Yes or No\nScamType: type_of_scam \nKeyFeaturesUsed: list of key features and their values\n\nDo not explain anything else."
        
        print(f"\n🔹 Q (Transaction CSV): {prompt.strip()}")
        clean_answer = self.llm_chat(prompt)
        print(f"🟢 A: {clean_answer}")
        
        full_log = f"Q: {prompt.strip()}\n\nA: {clean_answer}"
        
        return full_log, clean_answer

    def unified_analysis(self, contract_path: str, csv_path: str) -> str:
        output_log = ""

        output_log += "\n📊 Analyzing transaction data...\n"
        tx_log, tx_answer = self.analyze_transaction_csv(csv_path)
        output_log += f"{tx_log}\n"

        output_log += "\n🔍 Analyzing smart contract code...\n"
        contract_log, contract_answer = self.analyze_contract_code(contract_path)
        output_log += f"{contract_log}\n"

        final_summary_prompt = f"""
You are an experienced blockchain security auditor.

Here are the bimodal analysis results for a token:

[Transaction Feature Analysis]:
{tx_answer}

[Contract Analysis]:
{contract_answer}

Please synthesize the above results carefully. Use the following decision logic:

- If any modality indicates scam, output Scam: Yes and the scam type.
- If both modalities indicate no scam, output Scam: No.

Only provide the final answer in the following strict format:

Scam: Yes or No
ScamType: type_of_scam 

Do not provide any explanation.
"""
        
        print("\n🧠 Generating Final Assessment...")
        final_decision = self.llm_chat(final_summary_prompt)
        output_log += "\n🧠 Final Assessment:\n"
        output_log += f"{final_decision}\n"

        return output_log