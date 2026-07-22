import json
from pathlib import Path

def merge_json_results(directory_path, output_filename):
    directory = Path(directory_path)
    merged_data = {}
    
    # Files to merge
    files = [
        "results.json",
        "results_256_2048_ctx.json",
        "results_512_1024_ctx.json"
    ]
    
    for filename in files:
        file_path = directory / filename
        if not file_path.exists():
            print(f"Warning: {filename} not found.")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Merge dictionaries recursively
        for task, budgets in data.items():
            if task not in merged_data:
                merged_data[task] = {}
            for budget, policies in budgets.items():
                merged_data[task][budget] = policies
    
    # Sort budgets numerically for each task
    sorted_merged_data = {}
    for task, budgets in merged_data.items():
        # Sort keys by converting to int
        sorted_keys = sorted(budgets.keys(), key=lambda x: int(x))
        sorted_merged_data[task] = {k: budgets[k] for k in sorted_keys}
                
    # Save merged result
    output_path = directory / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_merged_data, f, indent=2)
    
    print(f"Successfully merged into {output_path}")

if __name__ == "__main__":
    merge_json_results(
        r"d:\MyFolder\ProgrammingWith-Python\Ai\A+\notebook\kv_cache_benchmark\039_swap_experiment",
        "merged_results.json"
    )
