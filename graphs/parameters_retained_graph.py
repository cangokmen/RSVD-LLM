import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Define ratios to analyze
ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

# Initialize data storage
data = {
    'ratio': [],
    'kept_params': []
}

# Read data from compression_summary.txt files
base_dir = '../benchmark_results_comparison'

for ratio in ratios:
    summary_file = os.path.join(base_dir, f'ratio_{ratio}', 'compression_summary.txt')
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            content = f.read()
            
        # Parse kept parameters percentage
        kept_params = None
        
        lines = content.split('\n')
        
        for line in lines:
            if 'Compression Ratio:' in line and 'keeping' in line:
                # Extract the decimal value (e.g., "0.9 or 90.0%")
                parts = line.split('keeping')[1].strip().split('or')
                if len(parts) >= 1:
                    # Extract just the decimal number before 'or', remove % if present
                    value_str = parts[0].strip().split()[0].replace('%', '')
                    kept_params = float(value_str) / 100 if float(value_str) > 1 else float(value_str)
        
        # Only append if value was found
        if kept_params is not None:
            data['ratio'].append(ratio)
            data['kept_params'].append(kept_params)
        else:
            print(f"Warning: Could not parse kept parameters from {summary_file}")
    else:
        print(f"Warning: {summary_file} not found")

# Assume LLaMA-7B has approximately 7 billion parameters
total_params = 7e9

# Convert fraction to actual number of parameters
kept_params_count = [kept * total_params for kept in data['kept_params']]

# Convert ratios to 1-ratio for x-axis
x_values = [1 - r for r in data['ratio']]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bar chart
bars = ax.bar(x_values, kept_params_count, width=0.06, color='#9b59b6', 
              label='Compressed Model Parameters', edgecolor='black', linewidth=1.2)

# Add horizontal line for base model
ax.axhline(y=total_params, color='#2ecc71', linestyle='--', linewidth=2, 
           label='Base Model (7B parameters)', zorder=0)

# Customize plot
ax.set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Parameters (billions)', fontsize=12, fontweight='bold')
ax.set_title('Number of Parameters', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3, axis='y')

# Set axis limits and ticks
ax.set_xticks([1 - r for r in ratios])
ax.set_ylim([0, total_params * 1.05])
# Format y-axis to show billions
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.1f}'))


plt.tight_layout()

# Save the plot
output_file = 'parameters_retained_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Graph saved as {output_file}")

plt.show()
