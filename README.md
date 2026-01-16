# <img src="./OctopusGuard.png" height="30" style="vertical-align: middle;"/> OctopusGuard: K-Line Enhanced Token Scam Detector Powered by Multimodal LLMs
## Overview

**OctopusGuard** is a unified multimodal token scam detector, integrating K-line (candlestick) chart visual patterns, on-chain transaction behavior, and smart contract code semantics. By leveraging large multimodal language models (LLMs) and retrieval-augmented generation (RAG) over a K-line vector database, OctopusGuard achieves state-of-the-art performance in detecting token scams, demonstrating superior accuracy and reasoning capabilities on our comprehensive benchmark dataset.

---

## 🔧 Dependency & Environment Setup

First, clone the repository and install all required dependencies.

```bash
cd OctopusGuard
pip install -r requirements.txt
```

---

## 📊 Data Sources & Structure

All core datasets are located under the `OctopusGuard/data/` directory:

- `experiment_address.csv`: A list of token contract addresses used for evaluation.
- `token_kline_images/`: Contains the K-line (candlestick) chart images for each token (one `.png` file per address).
- `token_transfer_files/`: Contains the on-chain transaction history for each token (one `.csv` file per address).
- `test_multimodal_data.json`: The main multimodal test set, which contains triplets of contracts, images, and transaction data for evaluation.

**Example Directory Tree:**
```
OctopusGuard/data/
  ├── experiment_address.csv
  ├── token_contracts/0x....sol
  ├── token_kline_images/0x....png
  ├── token_transfer_files/0x....csv
  └── test_multimodal_data.json
```

---

## 🔬 Evaluation Dataset
Our primary dataset, used for comprehensive evaluation, is located at:
*   **`data/test_multimodal_data.json`**

This benchmark dataset contains **300 unique tokens** that have been manually analyzed and labeled from multiple perspectives. For each token, we constructed a set of six distinct prompt-completion pairs to capture analysis across different modalities and reasoning steps. This structured approach allows for targeted model training and detailed evaluation of each analytical capability.

The structure for each token's six samples is designed to test the model's capabilities in a chained, multimodal fashion:

| Step | Modality / Task | Description |
| :--- | :--- | :--- |
| **1** | **K-Line Chart Analysis** | The model is given a K-line (candlestick) image and prompted to identify if it exhibits patterns indicative of a "Pump and Dump" scheme and if it's a scam. |
| **2** | **Transaction Feature Analysis** | The model receives a set of extracted on-chain transaction features (e.g., `creator_hold_ratio`, `receiver_entropy`) and is asked to classify the token as a scam, identify the scam type, and list the key features that led to its conclusion. |
| **3** | **Smart Contract Analysis (Part 1: Invariant Identification)** | Given the raw Solidity source code, the model is prompted to act as a static analysis tool and identify critical assertion invariants, outputting them in `line_number+ assert(expression);` format. |
| **4** | **Smart Contract Analysis (Part 2: Invariant Ranking)** | Using the source code and the previously generated invariants, the model is then asked to rank these invariants based on their security criticality. This tests deeper code understanding. |
| **5** | **Smart Contract Analysis (Part 3: Vulnerability Labeling)** | The model is prompted to perform a direct security audit on the source code and list any vulnerabilities found (e.g., `Honeypot`, `RugPull`). |
| **6** | **Multimodal Synthesis** | Finally, the model is presented with the summarized outputs from the previous analyses (K-line, Transaction, and Contract) and is tasked with acting as a senior auditor to synthesize this information into a final, definitive scam judgment. |

---

## 🛠️ Data Acquisition & Preprocessing

Scripts to gather and process data are located in `OctopusGuard/scripts/` and `OctopusGuard/data_processing/`.

### 1. Contract Code & Transaction Data

These scripts download the necessary on-chain data from BscScan.
- `scripts/data_acquisition/get_token_contract_code.py`: Downloads verified contract source code for all addresses in `experiment_address.csv`.
- `scripts/data_acquisition/get_token_transfer_files.py`: Crawls all token transfer transactions for each contract address.

**Usage:**
```bash
cd OctopusGuard/scripts/data_acquisition
python get_token_contract_code.py
python get_token_transfer_files.py
```

### 2. K-line Vector Database Construction

This script extracts visual embeddings from K-line images and builds a FAISS vector database for RAG.
- `data_processing/klines_rag_db/run_build_vector_db.py`

**Usage:**
```bash
cd OctopusGuard/data_processing/klines_rag_db
python run_build_vector_db.py
```

---

## 🚀 Reproducing Evaluations

This section describes how to reproduce the full evaluation results from the paper.

### 1. Multimodal Inference Pipelines

- **Llama-based pipeline:**  `OctopusGuard/src/OctopusGuardLlama.py`
- **Qwen-based pipeline:**   `OctopusGuard/src/OctopusGuardQwen.py`

Running these scripts will iterate through all tokens, perform the multimodal analysis, and log the results required for evaluation.

### 2. Evaluation & Ablation Analysis

All evaluation notebooks and logs are under `OctopusGuard/evaluations/`:
- `ablation_and_training_progress/`
- `backbone_comparison/`
- `competitor_benchmarking/`

**Example Usage:**
- To reproduce ablation and training progress results, see `ablation_and_training_progress/ablation_and_training_progress_evaluation.ipynb`.
- To compare different backbone models, see `backbone_comparison/backbone_comparison.ipynb`.
- To benchmark against other tools (e.g., Slither, HoneypotIs, SmartInv), see `competitor_benchmarking/competitor_benchmarking.ipynb`.

---

## 💻 Application

This section provides instructions for running the analysis on a single or multiple tokens using the pre-trained models.

### 1. Using the Qwen2.5-VL-7B-Instruct Backbone

- **Entry Point**: `src/OctopusGuardQwen.py`
- **How to Run**:
  1. Ensure all data directories (`data/token_contracts/`, `data/token_kline_images/`, `data/token_transfer_files/`) are populated for the tokens you wish to analyze.
  2. Execute the script:
     ```bash
     python src/OctopusGuardQwen.py
     ```
  3. The script will load the fine-tuned Qwen2.5-VL-7B-Instruct checkpoint, process each address, and output detailed inference logs to `evaluations/ablation_and_training_progress/logs/analysis_log_checkpoint-296.txt`.

### 2. Using the Llama-3.2-11B-Vision-Instruct Backbone

- **Entry Point**: `src/OctopusGuardLlama.py`
- **How to Run**:
  1. Ensure all data directories are populated as described above.
  2. Execute the script:
     ```bash
     python src/OctopusGuardLlama.py
     ```
  3. The script will load the fine-tuned Llama-3.2-11B-Vision-Instruct checkpoint, process each address, and output logs to `evaluations/backbone_comparison/analysis_log_Llama32Vision.txt`.