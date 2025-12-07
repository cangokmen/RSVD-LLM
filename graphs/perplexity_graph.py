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
    'original': [],
    'rsvd': [],
    'svd': []
}

# Read data from evaluation_summary.txt files
base_dir = '../benchmark_results_comparison'

for ratio in ratios:
    summary_file = os.path.join(base_dir, f'ratio_{ratio}', 'evaluation_summary.txt')
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            content = f.read()
            
        # Parse perplexity values
        original_ppl = None
        rsvd_ppl = None
        svd_ppl = None
        
        lines = content.split('\n')
        in_perplexity_section = False
        
        for line in lines:
            if 'PERPLEXITY' in line:
                in_perplexity_section = True
            elif line.strip() == '':
                in_perplexity_section = False
            
            if in_perplexity_section:
                if line.startswith('- Original:'):
                    original_ppl = float(line.split(':')[1].strip())
                elif line.startswith('- RSVD:'):
                    rsvd_ppl = float(line.split(':')[1].strip())
                elif line.startswith('- SVD:'):
                    svd_ppl = float(line.split(':')[1].strip())
        
        # Only append if all values were found
        if original_ppl is not None and rsvd_ppl is not None and svd_ppl is not None:
            data['ratio'].append(ratio)
            data['original'].append(original_ppl)
            data['rsvd'].append(rsvd_ppl)
            data['svd'].append(svd_ppl)
        else:
            print(f"Warning: Could not parse all perplexity values from {summary_file}")
    else:
        print(f"Warning: {summary_file} not found")

# Convert ratios to 1-ratio for x-axis
x_values = [1 - r for r in data['ratio']]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(x_values, data['original'], marker='o', linewidth=2, markersize=8, 
         label='Original', color='#2ecc71')
plt.plot(x_values, data['rsvd'], marker='s', linewidth=2, markersize=8, 
         label='RSVD', color='#e74c3c')
plt.plot(x_values, data['svd'], marker='^', linewidth=2, markersize=8, 
         label='SVD', color='#3498db')

# Customize plot
plt.xlabel('Compression Ratio', fontsize=12, fontweight='bold')
plt.ylabel('Perplexity (lower is better)', fontsize=12, fontweight='bold')
plt.title('Model Perplexity vs Compression Ratio', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)

# Set x-axis ticks
plt.xticks([1 - r for r in ratios])

plt.tight_layout()

# Save the plot
output_file = 'perplexity_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Graph saved as {output_file}")

plt.show()