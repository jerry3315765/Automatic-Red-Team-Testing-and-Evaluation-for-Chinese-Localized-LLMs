import os
import sys
import json
from datetime import datetime

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import yaml
from src.attacks.generator import generate_prompts
from src.models.local_llm import LocalLLM
from src.evaluation.judge import Judge

BASE_DIR = os.path.dirname(__file__)

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_intermediate_results(results, output_dir, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{name}_{timestamp}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {filepath}")
    return filepath

def main():
    print("Starting Traditional Chinese LLM Jailbreak Experiment V2...")
    
    # 1. Load Configurations
    config = load_config(os.path.join(BASE_DIR, 'config', 'config.yaml'))
    models_config = load_config(os.path.join(BASE_DIR, 'config', 'models.yaml'))
    prompts_config = load_config(os.path.join(BASE_DIR, 'config', 'prompts.yaml'))

    # Load additional intents from data/raw/intents.txt if it exists
    raw_intents_path = os.path.join(BASE_DIR, 'data', 'raw', 'intents.txt')
    if os.path.exists(raw_intents_path):
        with open(raw_intents_path, 'r', encoding='utf-8') as f:
            file_intents = [line.strip() for line in f if line.strip()]
        if 'intents' not in prompts_config:
            prompts_config['intents'] = []
        prompts_config['intents'].extend(file_intents)
        print(f"Loaded {len(file_intents)} additional intents from file.")

    # 2. Generate Attack Prompts
    print("Generating attack prompts...")
    prompts = generate_prompts(prompts_config)
    print(f"Generated {len(prompts)} prompts.")

    output_dir = os.path.join(BASE_DIR, config.get('output_dir', 'data/results'))
    os.makedirs(output_dir, exist_ok=True)

    judge = Judge(config)  # Initialize Judge

    # 3. Initialize Models & Run Inference
    all_results = []
    
    for model_cfg in models_config['models']:
        print(f"Processing model: {model_cfg['name']}")

        model_results = []
        
        if model_cfg['type'] == 'local':
            try:
                llm = LocalLLM(model_cfg)
                model_results = llm.generate(prompts)

                # Free up memory if possible
                del llm
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Failed to run model {model_cfg['name']}: {e}")
                continue
        elif model_cfg['type'] == 'api':
            print(f"API model execution for {model_cfg['name']} not yet implemented in main loop.")
            
        # 4. Evaluation Loop
        print(f"Evaluating results for {model_cfg['name']}...")
        for res in model_results:
            eval_result = judge.evaluate(res['prompt'], res['response'])
            res['evaluation'] = eval_result
        
        # Save per-model results
        save_intermediate_results(model_results, output_dir, f"results_{model_cfg['name']}")
        all_results.extend(model_results)

    # Save aggregated results
    if all_results:
        save_intermediate_results(all_results, output_dir, "final_results_aggregated")
        print("Experiment run complete.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
