import os
import time
import requests
import json
import pandas as pd
import re

# ========== CONFIGURATION ==========
API_KEY = 'YOUR_API_KEY'  # Replace with your own BscScan API Key
JSON_FILE = r'experiment_data\experiment_address.csv'  
SAVE_DIR = r'experiment_data\token_contracts'  
FAILED_LOG = os.path.join(SAVE_DIR, 'failed_contracts.txt')
os.makedirs(SAVE_DIR, exist_ok=True)

csv_file = JSON_FILE
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Input file not found: {csv_file}")
df = pd.read_csv(csv_file)
addresses = list({addr.strip().lower() for addr in df['contract_address'].dropna()})

def get_contract_source(address, retries=5, delay=1):
    for attempt in range(1, retries + 1):
        try:
            print(f"  Attempt {attempt} to fetch: {address}")
            url = (
                f"https://api.bscscan.com/api"
                f"?module=contract&action=getsourcecode"
                f"&address={address}"
                f"&apikey={API_KEY}"
            )
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data["status"] == "1" and data["result"]:
                return data["result"][0]["SourceCode"]
            else:
                print(f"  Failed to fetch ({data.get('message')}), status: {data.get('status')}")
        except Exception as e:
            print(f"  Exception: {e}")
        time.sleep(delay)
    return None

failed = []
for addr in addresses:
    filename = os.path.join(SAVE_DIR, f"{addr}.sol")
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        print(f"Exists, skipping: {addr}")
        continue
    print(f"\nDownloading: {addr}")
    source_code = get_contract_source(addr)
    if source_code:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(source_code)
        print(f"Saved: {addr}")
    else:
        print(f"Failed: {addr}")
        failed.append(addr)

if failed:
    with open(FAILED_LOG, 'w') as f:
        f.writelines(f"{addr}\n" for addr in failed)
    print(f"Failed addresses saved to: {FAILED_LOG}")
else:
    print("All contracts downloaded successfully!")

# Check which addresses in CSV do not have a public contract file
def check_missing_contracts(csv_path, contract_dir, output_path):
    df = pd.read_csv(csv_path)
    df['contract_address'] = df['contract_address'].str.lower().str.strip()
    csv_addresses = set(df['contract_address'])
    existing_files = os.listdir(contract_dir)
    existing_addresses = set(
        os.path.splitext(f)[0].lower()
        for f in existing_files
        if f.endswith('.sol')
    )
    missing_addresses = sorted(csv_addresses - existing_addresses)
    missing_df = pd.DataFrame(missing_addresses, columns=['contract_address_no_public'])
    missing_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Missing {len(missing_addresses)} addresses saved to: {output_path}")

# Fix contracts with JSON-wrapped source code
def try_fix_json(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("{{") and cleaned.endswith("}}"): cleaned = cleaned[1:-1]
    elif cleaned.startswith("{{"): cleaned = cleaned[1:]
    elif cleaned.endswith("}}"): cleaned = cleaned[:-1]
    return cleaned

def extract_content(raw_text):
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            fixed = try_fix_json(raw_text)
            data = json.loads(fixed)
        except json.JSONDecodeError:
            return None
    contents = []
    for v in data.values():
        if isinstance(v, dict) and "content" in v:
            contents.append(v["content"])
    if "sources" in data and isinstance(data["sources"], dict):
        for item in data["sources"].values():
            if "content" in item:
                contents.append(item["content"])
    if contents:
        return "\n\n".join(contents).strip()
    return None

def fix_contract_json(contract_dir):
    for fname in os.listdir(contract_dir):
        if not fname.endswith(".sol"):
            continue
        full_path = os.path.join(contract_dir, fname)
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if '"content"' in content:
            extracted = extract_content(content)
            if extracted:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(extracted)
                print(f"Fixed: {fname}")
            else:
                print(f"Could not extract source: {fname}")
        else:
            print(f"Normal contract: {fname}")

def aggressive_clean_code(source_code):
    lines = source_code.splitlines()
    cleaned = []
    prev_line = ""
    inside_function = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'function .*{', stripped):
            inside_function = True
            cleaned.append(line)
            prev_line = line
            continue
        if stripped == '}':
            inside_function = False
            if cleaned and cleaned[-1].strip() == '':
                cleaned.pop()
            cleaned.append(line)
            prev_line = line
            continue
        if stripped == '':
            if prev_line.strip().endswith("*/") or prev_line.strip().endswith("{"):
                continue
            if inside_function:
                continue
            if prev_line.strip() == '':
                continue
            cleaned.append('')
            prev_line = line
            continue
        cleaned.append(line)
        prev_line = line
    while cleaned and cleaned[-1].strip() == '':
        cleaned.pop()
    return '\n'.join(cleaned)

def batch_clean_solidity_files(target_dir):
    file_count = 0
    for filename in os.listdir(target_dir):
        if filename.endswith('.sol'):
            filepath = os.path.join(target_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            cleaned_content = aggressive_clean_code(content)
            if cleaned_content != content.strip():
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                print(f"Cleaned: {filename}")
                file_count += 1
            else:
                print(f"No cleaning needed: {filename}")
    print(f"Total cleaned files: {file_count}") 