import pandas as pd
import os
import ast
import numpy as np

# Configuration
RESULTS_ROOT = r"c:\Users\jerry\Desktop\lab\0311\PromptFuzz-Thesis\Results\focus\redteam\baseline"
OUTPUT_TEX = r"C:\Users\jerry\Desktop\lab\0311\thesis-experiment\paper\Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs\results_table.tex"

def list_string_to_sum(val):
    try:
        if isinstance(val, (int, float)): return val
        if isinstance(val, list): return sum(val)
        if isinstance(val, str):
            # Handle potential nested quotes or weird formatting
            val = val.strip()
            if val.startswith('[') and val.endswith(']'):
                li = ast.literal_eval(val)
                return sum(li)
        return 0
    except:
        return 0

def list_string_to_len(val):
    try:
        if isinstance(val, (int, float)): return 1
        if isinstance(val, list): return len(val)
        if isinstance(val, str):
            val = val.strip()
            if val.startswith('[') and val.endswith(']'):
                li = ast.literal_eval(val)
                return len(li)
        return 0
    except:
        return 0

def main():
    if not os.path.exists(RESULTS_ROOT):
        print("Results directory not found.")
        return

    model_stats = []

    for model_name in os.listdir(RESULTS_ROOT):
        model_path = os.path.join(RESULTS_ROOT, model_name)
        csv_path = os.path.join(model_path, "all_results.csv")
        
        if os.path.isdir(model_path) and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                # Calculate metrics
                df['success_count'] = df['results'].apply(list_string_to_sum)
                df['total_attempts'] = df['results'].apply(list_string_to_len)
                
                total_queries = len(df)
                total_attempts = df['total_attempts'].sum()
                total_successes = df['success_count'].sum()
                
                prompts_with_break = len(df[df['success_count'] > 0])
                
                esr = (prompts_with_break / total_queries * 100) if total_queries > 0 else 0
                asr = (total_successes / total_attempts * 100) if total_attempts > 0 else 0
                
                # Clean model name for LaTeX
                display_name = model_name.replace('_', '-').replace('@Q8-0', '').replace('-instruct', '').replace('-text', '').replace('-chat', '')
                
                model_stats.append({
                    "Model": display_name,
                    "Total Prompts": total_queries,
                    "ESR": esr,
                    "ASR": asr
                })
            except Exception as e:
                print(f"Skipping {model_name}: {e}")

    # Create DataFrame
    stats_df = pd.DataFrame(model_stats)
    
    # Sort by ASR
    stats_df = stats_df.sort_values('ASR', ascending=False)
    
    # Generate LaTeX Table
    latex_code = """
\\begin{table}[htbp]
\\centering
\\caption{Experimental Results: Jailbreak Success Rates on Localized vs. General LLMs}
\\label{tab:main_results}
\\begin{tabular}{l c c c}
\\toprule
\\textbf{Model} & \\textbf{Prompts Generated} & \\textbf{Prompt Coverage (ESR)} & \\textbf{True ASR} \\\\
\\midrule
"""
    
    for _, row in stats_df.iterrows():
        latex_code += f"{row['Model']} & {row['Total Prompts']} & {row['ESR']:.2f}\\% & {row['ASR']:.2f}\\% \\\\\n"
        
    latex_code += """\\bottomrule
\\end{tabular}
\\end{table}
"""

    print("Generated LaTeX Table:")
    print(latex_code)
    
    # Also save to file if needed or verify manually
    
    return latex_code

if __name__ == "__main__":
    main()
