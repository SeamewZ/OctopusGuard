import os
import pandas as pd
import time
import sys
from mutimodal_llama import unified_analysis

address_csv_path = "OctopusGuard/data/experiment_address.csv"
contract_df = pd.read_csv(address_csv_path)
contract_addresses = contract_df['contract_address'].tolist()

contract_dir = "OctopusGuard/data/token_contracts"
kline_dir = "OctopusGuard/data/token_kline_images"
transfer_dir = "OctopusGuard/data/token_transfer_files"

log_file = "logs/analysis_log_Llama32Vision.txt"
os.makedirs(os.path.dirname(log_file), exist_ok=True)

for idx, address in enumerate(contract_addresses):
    print(f"\n===== Analyzing the {idx+1}th address: {address} =====")

    norm_address = address.lower()

    contract_path = None
    image_path = None
    csv_path = None

    for file in os.listdir(contract_dir):
        if file.lower().startswith(norm_address):
            contract_path = os.path.join(contract_dir, file)
            break

    for file in os.listdir(kline_dir):
        if file.lower().startswith(norm_address):
            image_path = os.path.join(kline_dir, file)
            break

    for file in os.listdir(transfer_dir):
        if file.lower().startswith(norm_address):
            csv_path = os.path.join(transfer_dir, file)
            break

    if not contract_path or not image_path or not csv_path:
        print(f"⚠️ Missing files, skipping the address: {address}")
        continue

    try:
        result = unified_analysis(contract_path, csv_path, image_path)

        with open(log_file, "a", encoding="utf-8") as logf:
            logf.write(f"\n==================== Contract Address: {address} ====================\n")
            logf.write(result + "\n")
            logf.write(f"====================== ANALYSIS COMPLETE =====================\n\n")

    except Exception as e:
        print(f"❌ Analysis failed: {address}, error: {str(e)}")
        with open(log_file, "a", encoding="utf-8") as logf:
            logf.write(f"\n==================== Contract Address: {address} ====================\n")
            logf.write(f"Analysis failed, error: {str(e)}\n")
            logf.write(f"====================== Analysis failed =====================\n\n")

    time.sleep(2)

print("\n===== All addresses analyzed =====")