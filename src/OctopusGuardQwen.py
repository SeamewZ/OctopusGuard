import os
import pandas as pd
import time
import sys
import torch
sys.path.append('OctopusGuard/src')
from mutimodal_qwen_lib import Multimodal

BASE_PROJECT_PATH = "OctopusGuard/evaluations/ablation_and_training_progress"
BASE_MODEL_PATH = "OctopusGuard/src/qwen25vl_7b_finetune_checkpoint"

CLIP_PATH = "openai/clip-vit-base-patch32"
VECTOR_DB_PATH = "OctopusGuard/data_processing/klines_rag_db/vector_db/klines_faiss.index"
META_PATH = "OctopusGuard/data_processing/klines_rag_db/vector_db/metadata.pkl"

ADDRESS_CSV_PATH = "OctopusGuard/data/experiment_address.csv"
CONTRACT_DIR = "OctopusGuard/data/token_contracts"
KLINE_DIR = "OctopusGuard/data/token_kline_images"
TRANSFER_DIR = "OctopusGuard/data/token_transfer_files"

LOG_DIR = os.path.join(BASE_PROJECT_PATH, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

checkpoints_steps = [296]
try:
    contract_df = pd.read_csv(ADDRESS_CSV_PATH)
    contract_addresses = contract_df['contract_address'].tolist()
except FileNotFoundError:
    print(f"Error: Address file not found at {ADDRESS_CSV_PATH}")
    sys.exit(1)


for step in checkpoints_steps:
    model_id = os.path.join(BASE_MODEL_PATH, f"checkpoint-{step}")
    log_file = os.path.join(LOG_DIR, f"analysis_log_checkpoint-{step}.txt")

    print(f"\n{'='*20} STARTING EVALUATION FOR CHECKPOINT: checkpoint-{step} {'='*20}")
    print(f"Model Path: {model_id}")
    print(f"Log will be saved to: {log_file}")

    if not os.path.isdir(model_id):
        print(f"⚠️ Checkpoint directory not found, skipping: {model_id}")
        continue
    
    model = None 
    try:
        model = Multimodal(
            model_id=model_id,
            clip_path=CLIP_PATH,
            vector_db_path=VECTOR_DB_PATH,
            meta_path=META_PATH
        )

        for idx, address in enumerate(contract_addresses):
            print(f"\n----- Analyzing address {idx+1}/{len(contract_addresses)}: {address} (with checkpoint-{step}) -----")

            norm_address = address.lower()
            contract_path = next((os.path.join(CONTRACT_DIR, f) for f in os.listdir(CONTRACT_DIR) if f.lower().startswith(norm_address)), None)
            image_path = next((os.path.join(KLINE_DIR, f) for f in os.listdir(KLINE_DIR) if f.lower().startswith(norm_address)), None)
            csv_path = next((os.path.join(TRANSFER_DIR, f) for f in os.listdir(TRANSFER_DIR) if f.lower().startswith(norm_address)), None)

            if not all([contract_path, image_path, csv_path]):
                print(f"⚠️ Missing files for address, skipping: {address}")
                continue

            try:
                result = model.unified_analysis(contract_path, csv_path, image_path)

                with open(log_file, "a", encoding="utf-8") as logf:
                    logf.write(f"\n==================== Contract Address: {address} ====================\n")
                    logf.write(result + "\n")
                    logf.write("====================== ANALYSIS COMPLETE ======================\n\n")

            except Exception as e:
                print(f"❌ Analysis failed for address {address}: {str(e)}")
                with open(log_file, "a", encoding="utf-8") as logf:
                    logf.write(f"\n==================== Contract Address: {address} ====================\n")
                    logf.write(f"Analysis failed with error: {str(e)}\n")
                    logf.write("====================== ANALYSIS FAILED ======================\n\n")
            
            time.sleep(1) 

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to initialize or run for checkpoint-{step}. Error: {str(e)}")
        with open(log_file, "a", encoding="utf-8") as logf:
            logf.write(f"\nCRITICAL ERROR for checkpoint-{step}: {str(e)}\n")

    finally:
        if model:
            del model
        torch.cuda.empty_cache()
        print(f"GPU memory cleared for next run.")
        print(f"{'='*20} FINISHED EVALUATION FOR CHECKPOINT: checkpoint-{step} {'='*20}\n")
        time.sleep(5) 

print("\n===== ALL CHECKPOINTS EVALUATED =====")

