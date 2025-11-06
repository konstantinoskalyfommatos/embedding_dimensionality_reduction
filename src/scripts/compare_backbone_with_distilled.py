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


def collect_results_to_csv():
    """Collect all evaluation results and create a comparison CSV."""
    
    results_dir = os.path.join(PROJECT_ROOT, "storage", "results")
    
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
        for root, dirs, files in os.walk(model_dir):
            for result_file in files:
                if not result_file.endswith(".json") or result_file == "model_meta.json":
                    continue
                    
                task_name = os.path.splitext(result_file)[0]
                result_path = os.path.join(root, result_file)
                
                try:
                    with open(result_path, 'r') as f:
                        data = json.load(f)
                        
                    # Extract the main score from the test split
                    if "test" in data.get("scores", {}):
                        main_score = data["scores"]["test"][0].get("main_score")
                        
                        if task_name not in all_results:
                            all_results[task_name] = {}
                        
                        all_results[task_name][model_name] = main_score
                        
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Error reading {result_path}: {e}")
                    continue
    
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
    remaining_tasks = [task for task in df.index if task not in ordered_rows and not task.startswith("Average")]
    ordered_rows.extend(remaining_tasks)
    
    # Reorder the dataframe
    df = df.loc[ordered_rows]
    
    df = df.reindex(sorted(df.columns), axis=1)
    
    # Save to CSV
    output_path = os.path.join(PROJECT_ROOT, "storage", "comparison_results.csv")
    df.to_csv(output_path)

    md_output_path = os.path.join(PROJECT_ROOT, "storage", "comparison_results.md")
    df.to_markdown(md_output_path, floatfmt=".4f")
    
    print(f"Results saved to: {output_path}")
    print(f"\nSummary:")
    print(f"Tasks: {len(df)}")
    print(f"Models: {len(df.columns)}")
    print(f"\nPreview:\n{df.head(15)}")
    
    return df


if __name__ == "__main__":
    collect_results_to_csv()