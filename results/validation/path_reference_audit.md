# Path Reference Audit

- Scanned root: `C:\Users\jerry\Desktop\lab\0311\thesis-redteam-public`
- Target dirs: analysis, src, config
- Files with findings: 22

## analysis/analyze_cluster_shap_summary.py
- L15 [stale-hint] `experiment_v2`
- L18 [stale-hint] `experiment_v2`

## analysis/analyze_shap_clusters.py
- L189 [absolute-path] `s:\n`

## analysis/analyze_success_rates.py
- L10 [stale-hint] `experiment_v2`
- L134 [stale-hint] `experiment_v2`

## analysis/analyze_success_rates_v2.py
- L14 [stale-hint] `experiment_v2`
- L254 [stale-hint] `experiment_v2`

## analysis/audit_path_references.py
- L7 [absolute-path] `/Users/[^\`
- L7 [absolute-path] `/home/[^\`
- L9 [stale-hint] `thesis-experiment`
- L10 [stale-hint] `experiment_v2`
- L11 [stale-hint] `C:/Users`
- L12 [absolute-path] `C:\\\\Users`

## analysis/calc_paper_stats.py
- L8 [absolute-path] `c:\Users\jerry\Desktop\lab\experiment_v2\data\results`
- L8 [stale-hint] `experiment_v2`

## analysis/combine_ab_and_run_divi.py
- L12 [absolute-path] `c:\Users\jerry\Desktop\lab\0311`
- L13 [absolute-path] `c:\Users\jerry\Desktop\lab\0311\thesis-experiment\analysis`
- L13 [stale-hint] `thesis-experiment`
- L14 [absolute-path] `c:\Users\jerry\Desktop\lab\0311\Results`

## analysis/compute_experiment_metrics.py
- L13 [stale-hint] `experiment_v2`
- L18 [stale-hint] `experiment_v2`

## analysis/extract_all_keywords.py
- L9 [absolute-path] `c:\Users\jerry\Desktop\lab\0311\thesis-experiment\results\divi_combined`
- L9 [stale-hint] `thesis-experiment`

## analysis/generate_thesis_assets.py
- L12 [absolute-path] `c:\Users\jerry\Desktop\lab\0311\thesis-experiment\results\divi_combined`
- L12 [stale-hint] `thesis-experiment`
- L13 [absolute-path] `c:\Users\jerry\Desktop\lab\0311\thesis-experiment\paper\Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs\figures`
- L13 [stale-hint] `thesis-experiment`

## analysis/replot_figures.py
- L51 [absolute-path] `c:\Users\jerry\Desktop\lab\0311\thesis-experiment\paper\Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs\figures\comparison_asr.pdf`
- L51 [stale-hint] `thesis-experiment`

## analysis/update_paper_stats.py
- L7 [absolute-path] `/Users/jerry/Desktop/lab/experiment_v2/data/results/redteam_divi_results_*.json`
- L7 [stale-hint] `experiment_v2`
- L7 [stale-hint] `C:/Users`

## analysis/validate_paper_consistency.py
- L27 [stale-hint] `thesis-experiment`

## analysis/visualize_seed_results.py
- L10 [absolute-path] `c:\Users\jerry\Desktop\lab\0311\thesis-experiment\results\divi_combined`
- L10 [stale-hint] `thesis-experiment`
- L11 [absolute-path] `c:\Users\jerry\Desktop\lab\0311\thesis-experiment\paper\Automatic_Red_Team_Testing_and_Evaluation_for_Traditional_Chinese_Localized_LLMs\figures`
- L11 [stale-hint] `thesis-experiment`

## src/DIVI/DIVI.py
- L15 [absolute-path] `c:\Users\jerry\Desktop\lab\gibbs\gibbs_rework\full_embedding_result.csv`
- L16 [absolute-path] `c:\Users\jerry\Desktop\lab\gibbs\gibbs_rework`

## src/DIVI/DIVI_V2.py
- L15 [absolute-path] `C:\Users\jerry\Desktop\lab\code\merged_results_embedded.csv`
- L16 [absolute-path] `C:\Users\jerry\Desktop\lab\code\divi_results`

## src/convert_promptfuzz_to_traces.py
- L9 [stale-hint] `thesis-experiment`
- L90 [stale-hint] `thesis-experiment`

## src/convert_scenarios_to_promptfuzz.py
- L109 [stale-hint] `thesis-experiment`

## src/evaluation/re_evaluate_results.py
- L12 [stale-hint] `experiment_v2`

## src/run_promptfuzz_baseline.py
- L55 [stale-hint] `thesis-experiment`

## src/utils/merge_csv_results.py
- L9 [stale-hint] `experiment_v2`

## src/utils/setup_pipeline_structure.py
- L7 [stale-hint] `experiment_v2`
- L70 [stale-hint] `experiment_v2`

