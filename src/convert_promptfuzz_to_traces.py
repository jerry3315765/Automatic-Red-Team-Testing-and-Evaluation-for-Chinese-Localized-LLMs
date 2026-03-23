import json
import os
import pandas as pd
import argparse

def convert_promptfuzz_to_traces(results_csv, output_json, model_name="gemma-3-taide-12b-chat"):
    """
    Reads PromptFuzz output (from Results/focus/redteam/*.csv)
    and converts to thesis-experiment raw_traces format.
    """
    if not os.path.exists(results_csv):
        print(f"File not found: {results_csv}. Please run PromptFuzz first.")
        return

    df = pd.read_csv(results_csv)
    
    # Load Defense Mapping if available
    mapping_path = os.path.join(os.path.dirname(results_csv), "../../../../Datasets/redteam_focus_defense_mapping.json")
    # Resolve relative path
    mapping_path = os.path.abspath(mapping_path)
    
    defense_map = {}
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                defense_list = json.load(f)
                for d in defense_list:
                    defense_map[d['id']] = d['name']
        except:
            print("Warning: Could not load defense mapping.")

    traces = []
    for idx, row in df.iterrows():
        # PromptFuzz output columns: 'prompt', 'response', 'results'
        # 'response' and 'results' are string representation of lists
        prompt = row.get('prompt', str(row.get('attack', '')))
        
        try:
            responses = eval(row.get('response', '[]'))
            results = eval(row.get('results', '[]'))
        except:
            responses = [row.get('response', '')]
            results = [0] # Assume failure if parsing fails or simple format

        # Each prompt is tested against multiple defenses (scenarios).
        # We need to map them back if possible, or treat them as individual samples.
        # Since we don't have the explicit defense ID in the CSV row easily (it's implicit in the order),
        # We will generate one trace per response.
        
        for i, (resp, res) in enumerate(zip(responses, results)):
            is_success = bool(res)
            defense_name = defense_map.get(i, f"Defense_{i}")
            
            trace = {
                "scenario_id": idx, # Using row index as proxy for ID group
                "scenario_desc": f"PromptFuzz Mutated Prompt ({defense_name})",
                "attack_type": "promptfuzz",
                "system_prompt_name": defense_name,
                "system_prompt": "See defense mapping",
                "turn": 1,
                "prompt": prompt,
                "response": resp,
                "evaluation": {
                    "score": 5 if is_success else 1, 
                    "success": is_success,
                    "classification": "Jailbreak" if is_success else "Refusal",
                    "reason": "Evaluated by PromptFuzz Predictor"
                },
                "model": model_name,
                "success": is_success,
                "conversation_history": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp}
                ]
            }
            traces.append(trace)

    # Make sure output dir exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
        
    print(f"Converted {len(traces)} PrompFuzz results -> {output_json}")
    print("Next step: You can run src/integrate_redteam_divi.py or analyze_success_rates.py using this JSON.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="PromptFuzz-Thesis/Results/focus/redteam/0.csv")
    parser.add_argument("--out", type=str, default="thesis-experiment/results/raw_traces/redteam_divi_results_promptfuzz.json")
    parser.add_argument("--model", type=str, default="gemma-3-taide-12b-chat")
    args = parser.parse_args()
    
    convert_promptfuzz_to_traces(args.csv, args.out, args.model)
