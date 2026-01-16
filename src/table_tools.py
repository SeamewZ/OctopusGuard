import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict
from itertools import combinations

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df.dropna(subset=['amount'], inplace=True)
    return df

def detect_mint_address(df: pd.DataFrame) -> str:
    zero_address = '0x0000000000000000000000000000000000000000'
    mint_tx = df[df['sender'] == zero_address]
    if not mint_tx.empty:
        return mint_tx.iloc[0]['receiver']
    return df['receiver'].value_counts().idxmax()

def compute_creator_hold_ratio(df: pd.DataFrame, creator: str) -> float:
    """
    Calculate the net holding ratio of the creator (considering mint and burn transactions).
    """
    received = df[df['receiver'] == creator]['amount'].sum()
    sent = df[df['sender'] == creator]['amount'].sum()
    creator_net_hold = received - sent
    mint_amount = df[df['sender'] == '0x0000000000000000000000000000000000000000']['amount'].sum()
    burn_amount = df[df['receiver'] == '0x000000000000000000000000000000000000dead']['amount'].sum()
    total_supply = mint_amount - burn_amount
    if total_supply <= 0:
        return 0.0
    else:
        return round(creator_net_hold / total_supply, 6)

def compute_topn_ratio(df: pd.DataFrame, n: int = 10, exclude_addrs: set = None) -> float:
    """
    Calculate the holding ratio of the top N addresses to the total supply.
    """
    if exclude_addrs is None:
        exclude_addrs = set()
    received = df.groupby('receiver')['amount'].sum()
    sent = df.groupby('sender')['amount'].sum()
    net_hold = (received - sent).fillna(0)
    net_hold = net_hold[~net_hold.index.isin(exclude_addrs)]
    top_n_hold = net_hold.sort_values(ascending=False).head(n)
    top_n_total = top_n_hold.sum()
    mint_amount = df[df['sender'] == '0x0000000000000000000000000000000000000000']['amount'].sum()
    burn_amount = df[df['receiver'] == '0x000000000000000000000000000000000000dead']['amount'].sum()
    total_supply = mint_amount - burn_amount
    ratio = round(top_n_total / total_supply, 6) if total_supply > 0 else 0.0
    return ratio

def detect_blackhole_tx(df: pd.DataFrame) -> bool:
    return (df['receiver'] == '0x000000000000000000000000000000000000dead').any()

def detect_add_liquidity(df: pd.DataFrame) -> bool:
    for col in ['method', 'functionName']:
        if col in df.columns and df[col].str.lower().str.contains("addliquidity", na=False).any():
            return True
    return False

def get_method_distribution(df: pd.DataFrame) -> Dict[str, int]:
    return dict(Counter(df['method'].fillna('unknown').str.lower()))

def get_lifetime_seconds(df: pd.DataFrame) -> int:
    df['utc_time'] = pd.to_datetime(df['utc_time'], errors='coerce')
    return int((df['utc_time'].max() - df['utc_time'].min()).total_seconds())

def get_address_entropy(df: pd.DataFrame) -> float:
    # Calculate address entropy after removing airdrop and wash trading
    airdrop_tx = get_airdrop_transactions(df)
    wash_tx = get_wash_trading_transactions(df)
    filtered_df = df[~df.index.isin(airdrop_tx.index) & ~df.index.isin(wash_tx.index)]
    receivers = filtered_df['receiver'].value_counts(normalize=True)
    return round(-sum(p * np.log2(p) for p in receivers if p > 0), 4)

def get_transaction_summary(df: pd.DataFrame) -> Dict:
    total_tx = len(df)
    mint_amount = df[df['sender'] == '0x0000000000000000000000000000000000000000']['amount'].sum()
    burn_amount = df[df['receiver'] == '0x000000000000000000000000000000000000dead']['amount'].sum()
    total_supply = mint_amount - burn_amount
    unique_addresses = len(set(df['receiver']) | set(df['sender']))
    return {
        "total_transactions": total_tx,
        "total_supply": round(total_supply, 2),
        "unique_addresses": unique_addresses,
    }

def detect_airdrop_transactions_v3(df,
                                    min_batch_transfers=3,
                                    max_airdrop_amount=100,
                                    strict_txhash_mode=True):
    """
    More accurate airdrop detection logic (v3):
    - The same tx_hash appears multiple times
    - All records are from the same sender
    - At least min_batch_transfers different receivers
    - All amounts <= max_airdrop_amount
    - Exclude bidirectional transfers (sender and receiver overlap in the same tx)
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df['utc_time'] = pd.to_datetime(df['utc_time'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df[df['status'].str.upper() == 'SUCCESS']
    grouped = df.groupby('tx_hash')
    candidates = []
    for tx_hash, group in grouped:
        if len(group) < min_batch_transfers:
            continue
        if group['sender'].nunique() != 1:
            continue
        if group['receiver'].nunique() < min_batch_transfers:
            continue
        if group['amount'].max() > max_airdrop_amount:
            continue
        if set(group['sender']).intersection(set(group['receiver'])):
            continue
        candidates.append(group)
    return pd.concat(candidates).drop_duplicates() if candidates else pd.DataFrame(columns=df.columns)

def detect_wash_trading(df, threshold_interactions=10) -> pd.DataFrame:
    counter = Counter()
    for _, row in df.iterrows():
        addr_pair = tuple(sorted([row['sender'], row['receiver']]))
        counter[addr_pair] += 1
    suspicious_pairs = []
    for pair, count in counter.items():
        if count >= threshold_interactions:
            avg_amt = df[
                ((df['sender'] == pair[0]) & (df['receiver'] == pair[1])) |
                ((df['sender'] == pair[1]) & (df['receiver'] == pair[0]))
            ]['amount'].astype(float).mean()
            suspicious_pairs.append({
                'address_1': pair[0],
                'address_2': pair[1],
                'tx_count': count,
                'average_amount': avg_amt
            })
    return pd.DataFrame(suspicious_pairs)

def get_airdrop_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify airdrop transactions (returns full records)
    """
    airdrop_df = detect_airdrop_transactions_v3(df)
    return airdrop_df

def get_wash_trading_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify wash trading transactions (returns full records)
    """
    wash_df = detect_wash_trading(df)
    wash_pairs = set(
        tuple(sorted([row['address_1'], row['address_2']]))
        for _, row in wash_df.iterrows()
    )
    return df[df.apply(lambda row: tuple(sorted([row['sender'], row['receiver']])) in wash_pairs, axis=1)]

def compute_buy_sell_ratio(df: pd.DataFrame) -> Dict:
    try:
        senders = df['sender'].dropna().unique()
        receivers = df['receiver'].dropna().unique()
        all_users = set(senders) | set(receivers) 
        airdrop_tx = get_airdrop_transactions(df)
        wash_tx = get_wash_trading_transactions(df)
        filtered_df = df[~df.index.isin(airdrop_tx.index) & ~df.index.isin(wash_tx.index)]
        liquidity_pools = detect_liquidity_pool_addresses(df)
        buy_users = set(filtered_df[filtered_df['sender'].isin(liquidity_pools)]['receiver'])
        sell_users = set(filtered_df[filtered_df['receiver'].isin(liquidity_pools)]['sender'])
        return {
            "buy_user_count": len(buy_users),
            "sell_user_count": len(sell_users),
            "sell_user_ratio": round(len(sell_users) / len(all_users), 4) if len(all_users) else None,
            "buy_user_ratio": round(len(buy_users) / len(all_users), 4) if len(all_users) else None,
        }
    except Exception as e:
        print(f"compute_buy_sell_ratio exception: {e}")
        return {
            "buy_user_count": None,
            "sell_user_count": None,
            "sell_user_ratio": None,
            "buy_user_ratio": None
        }

def detect_slippage_loss(df: pd.DataFrame, threshold=0.3) -> Dict:
    """
    Detect user buy/sell amount ratio (slippage loss), based on liquidity pool addresses.
    """
    try:
        airdrop_tx = get_airdrop_transactions(df)
        wash_tx = get_wash_trading_transactions(df)
        filtered_df = df[~df.index.isin(airdrop_tx.index) & ~df.index.isin(wash_tx.index)].copy()
        filtered_df['amount'] = pd.to_numeric(filtered_df['amount'], errors='coerce')
        filtered_df.dropna(subset=['amount', 'sender', 'receiver'], inplace=True)
        pool_addresses = detect_liquidity_pool_addresses(filtered_df)
        user_stats = {}
        users_with_sell = set()
        for user in set(filtered_df['sender']).union(filtered_df['receiver']):
            buy_amount = filtered_df[(filtered_df['sender'].isin(pool_addresses)) & (filtered_df['receiver'] == user)]['amount'].sum()
            sell_amount = filtered_df[(filtered_df['receiver'].isin(pool_addresses)) & (filtered_df['sender'] == user)]['amount'].sum()
            if sell_amount > 0:
                users_with_sell.add(user)
            if buy_amount > 0 and sell_amount > 0:
                slippage = sell_amount / buy_amount
                user_stats[user] = slippage   
        low_slippage = [s for s in user_stats.values() if s < threshold]
        denominator = len(users_with_sell)
        return {
            "low_slippage_user_count": len(low_slippage),
            "low_slippage_users_ratio": round(len(low_slippage) / denominator, 4) if denominator else None
        }
    except Exception as e:
        print(f"detect_slippage_loss exception: {e}")
        return {
            "low_slippage_user_count": None,
            "low_slippage_users_ratio": None
        }

def detect_top_address_tx_ratio(df: pd.DataFrame, top_n: int = 10, exclude_addrs: set = None) -> dict:
    """
    Calculate the ratio of unique transactions participated by the top N most active addresses to the total number of transactions.
    """
    if exclude_addrs is None:
        exclude_addrs = set()
    address_involvement = pd.concat([df['sender'], df['receiver']]).value_counts()
    address_involvement = address_involvement[~address_involvement.index.isin(exclude_addrs)]
    top_addresses = set(address_involvement.head(top_n).index)
    mask = df['sender'].isin(top_addresses) | df['receiver'].isin(top_addresses)
    relevant_txs = df[mask]
    unique_tx_count = relevant_txs['tx_hash'].nunique() if 'tx_hash' in df.columns else len(relevant_txs)
    total_tx = len(df)
    top_tx_ratio = round(unique_tx_count / total_tx, 6) if total_tx > 0 else 0.0
    return {
        "top_addresses_tx_count": int(unique_tx_count),
        "top_addresses_tx_ratio": top_tx_ratio,
    }

def detect_liquidity_pool_addresses(df: pd.DataFrame) -> set:
    """
    Extract liquidity pool addresses from methods containing addLiquidity.
    """
    pool_addresses = set()
    fallback_cols = ['to', 'receiver', 'sender']
    for col in ['method', 'functionName']:
        if col in df.columns:
            mask = df[col].str.lower().str.contains("addliquidity", na=False)
            addr_col = next((fc for fc in fallback_cols if fc in df.columns), None)
            if addr_col:
                pool_addresses.update(df.loc[mask, addr_col].dropna().unique())
    return pool_addresses

def get_tx(file_path: str, K: int = 80) -> pd.DataFrame:
    df = load_csv(file_path)
    time_field = 'utc_time'
    df[time_field] = pd.to_datetime(df[time_field], format='%Y/%m/%d %H:%M:%S')
    earliest_k_df = df.sort_values(by=time_field).head(K)
    return earliest_k_df

def extract_transaction_features(file_path: str) -> dict:
    df = load_csv(file_path)
    creator = detect_mint_address(df)
    airdrop_transactions_df = detect_airdrop_transactions_v3(df)
    wash_trading_df = detect_wash_trading(df)
    features = {
        "creator_address": creator,
        "creator_hold_ratio": compute_creator_hold_ratio(df, creator),
        "blackhole_transfer": detect_blackhole_tx(df),
        "add_liquidity_called": detect_add_liquidity(df),
        "method_distribution": get_method_distribution(df),
        "lifetime_seconds": get_lifetime_seconds(df),
        "receiver_entropy": get_address_entropy(df),
        "airdrop_detected": not airdrop_transactions_df.empty,
        "airdrop_transactions_count": len(airdrop_transactions_df),
        "wash_trading_detected": not wash_trading_df.empty,
        "wash_trading_pairs_count": len(wash_trading_df),
        "wash_trading_trascation_nums": len(get_wash_trading_transactions(df)),
    }
    features.update(compute_buy_sell_ratio(df))
    features.update(detect_slippage_loss(df))
    liquidity_pool_addresses = detect_liquidity_pool_addresses(df)
    features.update(detect_top_address_tx_ratio(df, top_n=10, exclude_addrs=liquidity_pool_addresses))
    features.update(get_transaction_summary(df))
    return features
