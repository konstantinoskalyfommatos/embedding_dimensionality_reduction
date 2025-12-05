import os
import json
import pandas as pd
from utils.config import PROJECT_ROOT


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


def collect_results_to_df(results_dir: str):
    """Collect all evaluation results and create a comparison CSV."""
    
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    # Dictionary to store all results: {task_name: {model_name: score}}
    all_results = {}
    
    # Iterate through each model directory
    for model_name in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        # Navigate through the nested subdirectories to find JSON files
        for root, _, files in os.walk(model_dir):
            for result_file in files:
                if not result_file.endswith(".json") or result_file == "model_meta.json":
                    continue
                    
                task_name = os.path.splitext(result_file)[0]                
                with open(os.path.join(root, result_file), 'r') as f:
                    results = json.load(f)
                    
                test_scores = [subset['main_score'] for subset in results['scores']['test']]
                
                if task_name not in all_results:
                    all_results[task_name] = {}
                
                all_results[task_name][model_name] = sum(test_scores) / len(test_scores)
    
    if not all_results:
        print("No results found to compile.")
        return
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(all_results, orient='index')
    df.index.name = 'Task'
    
    # Reorder rows by task category and add averages
    ordered_rows = []
    for task_category, benchmarks in TASK_BENCHMARK_MAPPING.items():
        # Add individual benchmark rows
        for benchmark in benchmarks:
            if benchmark in df.index:
                ordered_rows.append(benchmark)
        
        # Calculate and add average row for this category
        category_benchmarks = [b for b in benchmarks if b in df.index]
        if category_benchmarks:
            avg_row = df.loc[category_benchmarks].mean()
            avg_row.name = f"**AVERAGE {task_category.upper()}**"
            df.loc[avg_row.name] = avg_row
            ordered_rows.append(avg_row.name)
    
    # Add any remaining tasks not in the mapping
    remaining_tasks = [
        task 
        for task in df.index 
        if task not in ordered_rows and not task.startswith("Average")
    ]
    ordered_rows.extend(remaining_tasks)
    
    # Reorder the dataframe
    df = df.loc[ordered_rows]
    df = df.reindex(sorted(df.columns), axis=1)
     
    return df


if __name__ == "__main__":
    results_dir = os.path.join(PROJECT_ROOT, "storage", "results")

    # TODO: Add arguments
    results_dir = "/home/kotsios/dsit/thesis/thesis_project/storage/evaluation_results/backbone/jinaai__jina-embeddings-v2-small-en"

    df = collect_results_to_df(results_dir=results_dir)

    # Save
    output_path = os.path.join(results_dir, "comparison_results.csv")
    df.to_csv(output_path)

    md_output_path = os.path.join(results_dir, "comparison_results.md")
    df.to_markdown(md_output_path, floatfmt=".4f")
    
    print(f"Results saved to: {output_path}")  

