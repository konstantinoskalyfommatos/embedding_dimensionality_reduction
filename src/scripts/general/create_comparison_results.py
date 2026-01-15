import os
import json
import pandas as pd
from argparse import ArgumentParser


TASK_BENCHMARK_MAPPING = {
    "sts": [
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STSBenchmark",
        "SICK-R"
    ],
    "retrieval": [
        "MIRACLRetrievalHardNegatives",
        "QuoraRetrievalHardNegatives",
        "HotpotQAHardNegatives",
        "DBPediaHardNegatives",
        "NQHardNegatives",
        "MSMARCOHardNegatives",
        "ArguAna"    
    ],
    "classification": [
        "AmazonCounterfactualClassification",
        "AmazonPolarityClassification",  
        "AmazonReviewsClassification",
        "ImdbClassification",  
        "ToxicConversationsClassification",        
    ],
    "clustering": [
        "ArxivClusteringS2S", 
        "RedditClustering",  
        "StackExchangeClustering"  
    ]
}


def collect_results_to_df(results_dir: str) -> pd.DataFrame:
    """Collect all evaluation results and create a comparison CSV."""
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    # Dictionary to store all results: {model_name: {task_name: score}}
    all_results = {}
    
    # Find all 'results' directories recursively
    for root, dirs, _ in os.walk(results_dir):
        if 'results' in dirs:
            results_subdir = os.path.join(root, 'results')
            
            # Inside each results directory, look for model directories
            for model_name in os.listdir(results_subdir):
                model_path = os.path.join(results_subdir, model_name)
                if not os.path.isdir(model_path):
                    continue
                
                # Initialize model entry if not exists
                if model_name not in all_results:
                    all_results[model_name] = {}
                
                # Walk through model directory to find JSON files
                for model_root, _, files in os.walk(model_path):
                    for result_file in files:
                        if not result_file.endswith(".json") or result_file == "model_meta.json":
                            continue
                        
                        task_name = os.path.splitext(result_file)[0]
                        
                        # Only store if not already present (avoid duplicates)
                        if task_name not in all_results[model_name]:
                            with open(os.path.join(model_root, result_file), 'r') as f:
                                results = json.load(f)
                                test_scores = [subset['main_score'] for subset in results['scores']['test']]
                                all_results[model_name][task_name] = sum(test_scores) / len(test_scores)
    
    if not all_results:
        print("No results found to compile.")
        return
    
    df = pd.DataFrame.from_dict(all_results, orient='index')
    df.index.name = 'Model'
    
    # Reorder columns by task category and add average columns
    ordered_cols = []
    avg_cols = []
    
    for task_category, benchmarks in TASK_BENCHMARK_MAPPING.items():
        # Add individual benchmark columns
        for benchmark in benchmarks:
            if benchmark in df.columns:
                ordered_cols.append(benchmark)
        
        # Calculate and add average column for this category
        category_benchmarks = [b for b in benchmarks if b in df.columns]
        if category_benchmarks:
            avg_col_name = f"**AVG_{task_category.upper()}**"
            df[avg_col_name] = df[category_benchmarks].mean(axis=1)
            avg_cols.append(avg_col_name)
    
    # Add any remaining tasks not in the mapping
    remaining_tasks = [
        task 
        for task in df.columns 
        if task not in ordered_cols and not task.startswith("**AVG_")
    ]
    ordered_cols.extend(remaining_tasks)
    
    # Calculate overall average across all tasks (excluding AVG columns)
    all_task_cols = ordered_cols
    df["**AVG_OVERALL**"] = df[all_task_cols].mean(axis=1)
    
    # Reorder the dataframe columns: overall avg first, then category avgs, then the rest
    df = df[["**AVG_OVERALL**"] + avg_cols + ordered_cols]
    df = df.sort_index()
     
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--results_base_path", 
        type=str, 
        default="storage/evaluation_results", 
        help="A directory that contains as subdirs result directories for multiple models"
    )

    args = parser.parse_args()
    df = collect_results_to_df(
        results_dir=args.results_base_path,
    )

    output_path = os.path.join(args.results_base_path, "comparison_results.csv")
    df.to_csv(output_path)
    
    print(f"Results saved to: {output_path}")
