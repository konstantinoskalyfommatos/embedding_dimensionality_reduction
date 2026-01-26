import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.config import EVALUATION_RESULTS_PATH, PROJECT_ROOT


def extract_embedding_dim(model_name):
    """Extract embedding dimension from model name."""
    if '_distilled_' in model_name:
        parts = model_name.split('_distilled_')
        dim_part = parts[1].split('_')[0]
        return int(dim_part)
    # Assume base models have their original dimensions
    elif 'gte-multilingual-base' in model_name:
        return 768  # GTE base dimension
    elif 'jina-embeddings-v2-small-en' in model_name:
        return 512  # Jina v2 small dimension
    return None

def extract_method(model_name):
    """Extract distillation method from model name."""
    if 'batch' in model_name:
        return 'custom'
    elif '_pca' in model_name:
        return 'pca'
    elif 'random_projection' in model_name:
        return 'random_projection'
    elif 'random_selection' in model_name:
        return 'random_selection'
    elif 'truncation' in model_name:
        return 'truncation'
    return 'base'

def plot_model_performance(
    model_base_name, 
    df, 
    output_dir,
    task_columns
):
    """Create performance plots for a specific model with different methods."""
    # Filter rows for this model
    model_data = df[df['Model'].str.contains(model_base_name.replace('/', '__'))]
    
    # Extract dimensions and methods
    model_data = model_data.copy()
    model_data['dimension'] = model_data['Model'].apply(extract_embedding_dim)
    model_data['method'] = model_data['Model'].apply(extract_method)
    model_data = model_data.dropna(subset=['dimension'])
    
    if len(model_data) == 0:
        print(f"No data found for model: {model_base_name}")
        return
    
    # Get all unique dimensions for this model
    all_dimensions = sorted(model_data['dimension'].unique())
    
    # Create plots for each task
    fig, axes = plt.subplots(2, 3, figsize=(30, 12))
    fig.suptitle(f'Performance vs Embedding Dimension: {model_base_name}', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    # Define colors and markers for each method
    colors = {'custom': 'blue', 'pca': 'red', 'random_projection': 'green', 
              'random_selection': 'orange', 'truncation': 'purple', 'base': 'black'}
    markers = {'custom': 'o', 'pca': 's', 'random_projection': '^', 
               'random_selection': 'D', 'truncation': 'v', 'base': '*'}
    
    for idx, (task_name, column_name) in enumerate(task_columns.items()):
        ax = axes[idx]
        
        # Collect all values for this task to determine y-axis range
        task_values = []
        
        # Plot each method separately
        for method_key, method_name in distillation_methods.items():
            method_data = model_data[model_data['method'] == method_key].sort_values('dimension')
            
            if len(method_data) > 0:
                ax.plot(method_data['dimension'], method_data[column_name], 
                       marker=markers[method_key], linewidth=2, markersize=8, 
                       label=method_name, color=colors[method_key], alpha=0.7)
                task_values.extend(method_data[column_name].tolist())
        
        # Plot base model as a horizontal line if it exists
        base_data = model_data[model_data['method'] == 'base']
        if len(base_data) > 0:
            base_score = base_data[column_name].values[0]
            base_dim = base_data['dimension'].values[0]
            ax.axhline(y=base_score, color=colors['base'], linestyle='--', 
                      linewidth=2, label=f'Base Model (dim={int(base_dim)})', alpha=0.7)
            ax.plot(base_dim, base_score, marker=markers['base'], 
                   markersize=12, color=colors['base'])
            task_values.append(base_score)
        
        ax.set_xlabel('Embedding Dimension', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{task_name} Method Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        
        # Set x-axis ticks to actual dimensions
        ax.set_xticks(all_dimensions)
        ax.set_xticklabels([int(d) for d in all_dimensions])
        # Stretch x-axis while preserving linear ratios between dimensions
        ax.set_xlim(all_dimensions[0], all_dimensions[-1])
        ax.margins(x=0)
        
        # Set y-axis limits with padding
        if task_values:
            y_min, y_max = min(task_values), max(task_values)
            padding = (y_max - y_min) * 0.1
            ax.set_ylim([max(0, y_min - padding), min(1, y_max + padding)])
    
    # Remove the extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    
    # Save the plot
    safe_model_name = model_base_name.replace('/', '_')
    output_path = os.path.join(output_dir, f'{safe_model_name}_method_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def create_method_comparison_plots(
    df, 
    output_dir,
    task_columns,
    models
):
    """Create separate plots comparing methods for each task, per model."""
    colors = {'custom': 'blue', 'pca': 'red', 'random_projection': 'green', 
              'random_selection': 'orange', 'truncation': 'purple', 'base': 'black'}
    markers = {'custom': 'o', 'pca': 's', 'random_projection': '^', 
               'random_selection': 'D', 'truncation': 'v', 'base': '*'}
    
    for model_base_name in models:
        model_short = model_base_name.split('/')[-1]
        model_data = df[df['Model'].str.contains(model_base_name.replace('/', '__'))]
        model_data = model_data.copy()
        model_data['dimension'] = model_data['Model'].apply(extract_embedding_dim)
        model_data['method'] = model_data['Model'].apply(extract_method)
        model_data = model_data.dropna(subset=['dimension'])
        
        if len(model_data) == 0:
            print(f"No data found for model: {model_base_name}")
            continue
        
        all_dimensions = sorted(model_data['dimension'].unique())
        
        for task_name, column_name in task_columns.items():
            fig, ax = plt.subplots(figsize=(20, 8))
            fig.suptitle(f'{task_name} Performance: {model_base_name}', fontsize=16, fontweight='bold')
            
            task_values = []
            
            # Plot base model as horizontal reference line
            base_data = model_data[model_data['method'] == 'base']
            if len(base_data) > 0:
                base_score = base_data[column_name].values[0]
                base_dim = base_data['dimension'].values[0]
                ax.axhline(y=base_score, color=colors['base'], linestyle='--', 
                          linewidth=2, label=f'Base Model (dim={int(base_dim)})', alpha=0.7)
                ax.plot(base_dim, base_score, marker=markers['base'], 
                       markersize=12, color=colors['base'])
                task_values.append(base_score)
            
            for method_key, method_name in distillation_methods.items():
                method_data = model_data[model_data['method'] == method_key].sort_values('dimension')
                
                if len(method_data) > 0:
                    ax.plot(method_data['dimension'], method_data[column_name], 
                           marker=markers[method_key], linewidth=2.5, markersize=8, 
                           label=method_name, color=colors[method_key], alpha=0.8)
                    task_values.extend(method_data[column_name].tolist())
            
            ax.set_xlabel('Embedding Dimension', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')
            
            # Set x-axis ticks to actual dimensions
            ax.set_xticks(all_dimensions)
            ax.set_xticklabels([int(d) for d in all_dimensions])
            # Stretch x-axis while preserving linear ratios between dimensions
            ax.set_xlim(all_dimensions[0], all_dimensions[-1])
            ax.margins(x=0)
            
            # Set y-axis limits with padding
            if task_values:
                y_min, y_max = min(task_values), max(task_values)
                padding = (y_max - y_min) * 0.1
                ax.set_ylim([max(0, y_min - padding), min(1, y_max + padding)])
            
            plt.tight_layout()
            
            safe_model_name = model_base_name.replace('/', '_')
            output_path = os.path.join(output_dir, f'{safe_model_name}_{task_name.lower()}_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved {task_name} comparison plot for {model_short}: {output_path}")
            plt.close()

if __name__ == "__main__":
    print("Creating performance plots")

    RESULTS_PATH = os.path.join(EVALUATION_RESULTS_PATH, "comparison_results.csv")
    PLOTS_PATH = os.path.join(PROJECT_ROOT, "storage/plots")

    os.makedirs(PLOTS_PATH, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)

    # Define the models to analyze
    models = [
        "Alibaba-NLP/gte-multilingual-base",
        "jinaai/jina-embeddings-v2-small-en"
    ]

    # Define task columns
    task_columns = {
        'Overall': '**AVG_OVERALL**',
        'STS': '**AVG_STS**',
        'Retrieval': '**AVG_RETRIEVAL**',
        'Classification': '**AVG_CLASSIFICATION**',
        'Clustering': '**AVG_CLUSTERING**'
    }

    # Define distillation methods and their display names
    distillation_methods = {
        'custom': 'Distillation (Ours)',
        'pca': 'PCA',
        'random_projection': 'Random Projection',
        'random_selection': 'Random Selection',
        'truncation': 'Truncation'
    }
    
    # Create individual plots for each model
    for model in models:
        plot_model_performance(model, df, PLOTS_PATH, task_columns)
    
    # Create method comparison plots for each task
    create_method_comparison_plots(df, PLOTS_PATH, task_columns, models)
    
    print("All plots created successfully!")