import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configuration
RESULTS_ROOT = r"C:\Users\jerry\Desktop\lab\0311\PromptFuzz-Thesis\Results\focus\redteam\baseline"
SCENARIO_CONFIG = r"C:\Users\jerry\Desktop\lab\0311\thesis-experiment\config\red_team_scenarios.json"
OUTPUT_DIR = r"C:\Users\jerry\Desktop\lab\0311\PromptFuzz-Thesis\Analysis_Report"

def load_scenarios():
    with open(SCENARIO_CONFIG, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Create mapping: index -> scenario_id & description
    # Assuming 5 seeds per scenario in focus_seed.jsonl
    # Index 0-4 -> Scenario 1, 5-9 -> Scenario 2...
    mapping = {}
    for i in range(50): # 10 scenarios * 5 seeds
        scenario_idx = i // 5
        if scenario_idx < len(data['scenarios']):
            s = data['scenarios'][scenario_idx]
            mapping[str(i)] = {
                "id": s['id'],
                "domain": s.get('description', '').split('：')[0] # Get text before colon as domain
            }
    return mapping

def list_string_to_sum(val):
    """Parses string representation of list "[1, 0, 1]" to sum."""
    try:
        # Check if it's already a list or number
        if isinstance(val, list):
            return sum(val)
        if isinstance(val, (int, float)):
            return val
            
        # Parse string "[1, 0, 1]"
        clean_val = val.strip('[]').replace(' ', '')
        if not clean_val: return 0
        items = clean_val.split(',')
        return sum([int(float(x)) for x in items])
    except:
        return 0

def analyze_all_models():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    scenario_map = load_scenarios()
    all_data = []

    # 1. Walk through all model folders
    if not os.path.exists(RESULTS_ROOT):
        print(f"Results directory not found: {RESULTS_ROOT}")
        return

    for model_name in os.listdir(RESULTS_ROOT):
        model_path = os.path.join(RESULTS_ROOT, model_name)
        csv_path = os.path.join(model_path, "all_results.csv")
        
        if os.path.isdir(model_path) and os.path.exists(csv_path):
            print(f"Analyzing {model_name}...")
            try:
                df = pd.read_csv(csv_path)
                
                # Basic cleaning
                df['model'] = model_name
                
                # Parse results column
                # Logic: If sum(results) > 0, consider it at least partially successful
                # Or you might want strict success (depends on your definition)
                df['jailbreak_score'] = df['results'].apply(list_string_to_sum)
                df['is_jailbreak'] = df['jailbreak_score'] > 0 
                
                # Map Parent Seed to Scenario
                # Note: 'parent_index' isn't explicitly in standard CSV sometimes, 
                # but 'parent' usually refers to the prompt node object. 
                # We need to trace back. 
                # Simplification: If the 'parent' column contains the index of the parent,
                # we can try to infer. 
                # However, for 'focus' phase, usually the first 50 rows ARE the seeds (if recorded)
                # or the first generated children come from seeds 0-49.
                # Let's assume we can't easily map dynamic nodes to scenario without a proper graph.
                # BUT, we can try to guess based on keyword matching if graph is missing.
                # For now, let's just do Model-Level stats.
                
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {model_name}: {e}")

    if not all_data:
        print("No data found.")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # --- Analysis 1: Overall ASR per Model ---
    # Metric 1: Attack Success Rate (True ASR) = Total Successful Responses / Total Queries (Responses)
    # Metric 2: Ensemble Success Rate (ESR) = Percentage of prompts that broke at least one defense
    
    full_df['success_count'] = full_df['results'].apply(lambda x: sum(list_string_to_sum(x)) if isinstance(list_string_to_sum(x), list) else list_string_to_sum(x))
    
    # Need to robustly handle the 'results' string again
    def robust_sum(val):
        try:
            if isinstance(val, (int, float)): return val
            import ast
            if isinstance(val, str):
                li = ast.literal_eval(val)
                return sum(li)
            if isinstance(val, list): return sum(val)
            return 0
        except: return 0

    def robust_len(val):
        try:
            if isinstance(val, (int, float)): return 1
            import ast
            if isinstance(val, str):
                li = ast.literal_eval(val)
                return len(li)
            if isinstance(val, list): return len(val)
            return 0
        except: return 0

    full_df['success_count'] = full_df['results'].apply(robust_sum)
    full_df['total_attempts'] = full_df['results'].apply(robust_len)
    full_df['is_jailbreak'] = full_df['success_count'] > 0

    summary = full_df.groupby('model').agg(
        Total_Prompts=('is_jailbreak', 'count'),
        Prompts_with_at_least_one_break=('is_jailbreak', 'sum'),
        Total_Attempts=('total_attempts', 'sum'),
        Total_Successes=('success_count', 'sum')
    ).reset_index()

    summary.rename(columns={'model': 'Model'}, inplace=True)

    summary['Prompt_ASR_ESR'] = (summary['Prompts_with_at_least_one_break'] / summary['Total_Prompts']) * 100
    summary['True_ASR'] = (summary['Total_Successes'] / summary['Total_Attempts']) * 100
    
    print("\n=== Overall ASR ===")
    print(summary[['Model', 'Prompt_ASR_ESR', 'True_ASR']])
    summary.to_csv(os.path.join(OUTPUT_DIR, "overall_asr.csv"), index=False)
    
    # Plot True ASR
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x='Model', y='True_ASR')
    plt.title('True Attack Success Rate (Total Successes / Total Attempts)')
    plt.ylabel('ASR (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "asr_by_model.png"))
    plt.close()

    # Plot Prompt ASR (ESR)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x='Model', y='Prompt_ASR_ESR')
    plt.title('Prompt Success Rate (Prompts finding >=1 Jailbreak)')
    plt.ylabel('ESR (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "esr_by_model.png"))
    plt.close()

    # --- Analysis 2: Mutator Effectiveness ---
    # Check if 'mutation' column exists
    if 'mutation' in full_df.columns:
        mutator_stats = full_df.groupby(['model', 'mutation'])['is_jailbreak'].mean().reset_index()
        
        # Pivot for heatmap
        pivot_table = mutator_stats.pivot(index='mutation', columns='model', values='is_jailbreak')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.2%')
        plt.title('Jailbreak Success Rate by Mutation Strategy')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "mutator_heatmap.png"))
        plt.close()
    
    print(f"\nAnalysis complete. Reports saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_all_models()
