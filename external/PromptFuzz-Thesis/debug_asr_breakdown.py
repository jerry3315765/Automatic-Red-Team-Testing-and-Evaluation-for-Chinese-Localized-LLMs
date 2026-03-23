import pandas as pd
import ast
import os

# Use relative path assuming we're in PromptFuzz-Thesis
csv_path = r"Results\focus\redteam\baseline\gpt-oss-20b\all_results.csv"

if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    # Try another one
    csv_path = r"Results/focus/redteam/baseline/gpt-oss-20b/all_results.csv"

def parse_results(val):
    try:
        if pd.isna(val): return []
        # Handle string representation of list "[0, 1, ...]"
        # If it's already a list, fine.
        if isinstance(val, list): return val
        if isinstance(val, str):
            return ast.literal_eval(val)
        return []
    except:
        return []

try:
    df = pd.read_csv(csv_path)
    # Filter rows that have meaningful prompts
    
    df['parsed_results'] = df['results'].apply(parse_results)
    
    # Calculate sum of successes per row
    df['success_count'] = df['parsed_results'].apply(sum)
    df['total_defenses'] = df['parsed_results'].apply(len)
    
    total_rows = len(df)
    success_rows = len(df[df['success_count'] > 0])
    failed_rows = len(df[df['success_count'] == 0])
    
    print(f"Total Rows (Prompts): {total_rows}")
    print(f"Prompts with > 0 successes: {success_rows} ({success_rows/total_rows:.2%})")
    print(f"Prompts with 0 successes: {failed_rows} ({failed_rows/total_rows:.2%})")
    
    total_attacks = df['total_defenses'].sum()
    total_successes = df['success_count'].sum()
    
    if total_attacks > 0:
        print(f"\n--- Detailed Breakdown ---")
        print(f"Total Individual Attacks (Prompts * Defenses): {total_attacks}")
        print(f"Total Individual Successes: {total_successes}")
        print(f"True ASR (Successes / Attacks): {total_successes/total_attacks:.4%}")
    else:
        print("No attacks found (empty results lists?)")

except Exception as e:
    print(f"Error: {e}")
