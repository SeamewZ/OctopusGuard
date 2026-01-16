import requests
import csv
import os
import time

# ========== CONFIGURATION ==========
API_KEY = 'YOUR_API_KEY' 
BASE_URL = 'https://api.bscscan.com/api'
input_csv = r"experiment_data\experiment_address.csv"  
output_dir = r"experiment_data\token_transfer_files" 

offset = 1000  # Max transactions per request (BSCScan limit)
WINDOW_MAX_RECORDS = 10000  # Max records per window
max_records = 49000  # Prevent excessive crawling per contract
sleep_time = 0.2  # API rate limit (5 req/sec for free API)
retry_max = 10
retry_sleep = 2
# ===================================

os.makedirs(output_dir, exist_ok=True)

try:
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        contract_addresses = [row['contract_address'] for row in reader]
except FileNotFoundError:
    print(f"Input file not found: {input_csv}")
    exit()

for idx, contract in enumerate(contract_addresses):
    print(f"\n{'='*50}")
    print(f"Processing contract {idx + 1}/{len(contract_addresses)}: {contract}")
    print(f"{'='*50}")

    output_path = os.path.join(output_dir, f"{contract}.csv")
    if os.path.exists(output_path):
        print(f"File already exists, skipping: {output_path}")
        continue

    # Find the earliest transaction block number
    params_first = {
        'module': 'account',
        'action': 'tokentx',
        'contractaddress': contract,
        'page': 1,
        'offset': 1,
        'sort': 'asc',
        'apikey': API_KEY
    }
    first_block = 0
    success = False
    for attempt in range(retry_max):
        try:
            response = requests.get(BASE_URL, params=params_first, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == '1' and data.get('result'):
                first_block = int(data['result'][0]['blockNumber'])
                print(f"Earliest transaction block for contract {contract}: {first_block}")
                success = True
                break
            else:
                print(f"No transaction records found for contract {contract}, skipping. Reason: {data.get('message', 'No result')}")
                success = False
                break
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}/{retry_max}): {e}")
            time.sleep(retry_sleep)

    if not success:
        continue

    # Sliding window + pagination crawling logic
    all_results = []
    seen_tx_signatures = set()
    
    start_block = first_block

    # Outer loop: sliding window
    while True:
        print(f"\nStarting new query window, start block: {start_block}")
        
        window_results = []
        is_window_full = False
        
        # Inner loop: pagination within current window
        page = 1
        while True:
            params = {
                'module': 'account',
                'action': 'tokentx',
                'contractaddress': contract,
                'startblock': start_block,
                'endblock': 999999999,
                'page': page,
                'offset': offset,
                'sort': 'asc',
                'apikey': API_KEY
            }

            result_this_page = None
            fetch_success = False
            for attempt in range(retry_max):
                try:
                    response = requests.get(BASE_URL, params=params, timeout=20)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data.get('status') == '1' and data.get('result'):
                        result_this_page = data['result']
                    else:
                        result_this_page = []
                    
                    fetch_success = True
                    break
                except requests.exceptions.RequestException as e:
                    print(f"  - Page {page} request failed (attempt {attempt + 1}/{retry_max}): {e}")
                    time.sleep(retry_sleep)
            
            if not fetch_success:
                print(f"  - Page {page} request failed, aborting current contract.")
                window_results = []
                break

            if not result_this_page:
                print(f"  -> No data on page {page}, window finished.")
                break
            
            window_results.extend(result_this_page)
            print(f"  -> Page {page}: Retrieved {len(result_this_page)} records. Window total: {len(window_results)}.")

            if len(result_this_page) < offset:
                break

            if len(window_results) >= WINDOW_MAX_RECORDS:
                is_window_full = True
                break
                
            page += 1
            time.sleep(sleep_time)
            

        if not window_results:
            print("All transaction records retrieved.")
            break

        # Deduplication logic
        newly_added_count = 0
        for tx in window_results:
            tx_signature = tuple(sorted(tx.items()))
            
            if tx_signature not in seen_tx_signatures:
                all_results.append(tx)
                seen_tx_signatures.add(tx_signature)
                newly_added_count += 1
        
        print(f"  -> Window finished. Newly added (deduplicated): {newly_added_count}. Total: {len(all_results)}.")

        if len(all_results) >= max_records:
            print(f"Reached or exceeded max record limit ({max_records}), stopping crawl.")
            all_results = all_results[:max_records]
            break
        
        if is_window_full:
            last_tx_in_window = window_results[-1]
            start_block = int(last_tx_in_window['blockNumber'])
            print(f"  -> Window full ({WINDOW_MAX_RECORDS}), moving to next start block: {start_block}")
        else:
            print("Current window not full, this is the last batch.")
            break

    # 3. Save results
    if all_results:
        fieldnames = all_results[0].keys()
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSaved {len(all_results)} transaction records for contract {contract} to {output_path}")
    elif not os.path.exists(output_path):
        print(f"No valid transaction records to save for contract {contract}.")

print("\nAll contracts processed.")