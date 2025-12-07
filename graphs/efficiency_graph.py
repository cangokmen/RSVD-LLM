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
            
        # Parse efficiency values (throughput)
        original_eff = None
        rsvd_eff = None
        svd_eff = None
        
        lines = content.split('\n')
        in_efficiency_section = False
        
        for line in lines:
            if 'EFFICIENCY' in line:
                in_efficiency_section = True
            elif line.strip() == '':
                in_efficiency_section = False
            
            if in_efficiency_section:
                if line.startswith('- Original:') and 'Throughput:' in line:
                    original_eff = float(line.split('Throughput:')[1].strip().split()[0])
                elif line.startswith('- RSVD:') and 'Throughput:' in line:
                    rsvd_eff = float(line.split('Throughput:')[1].strip().split()[0])
                elif line.startswith('- SVD:') and 'Throughput:' in line:
                    svd_eff = float(line.split('Throughput:')[1].strip().split()[0])
        
        # Only append if all values were found
        if original_eff is not None and rsvd_eff is not None and svd_eff is not None:
            data['ratio'].append(ratio)
            data['original'].append(original_eff)
            data['rsvd'].append(rsvd_eff)
            data['svd'].append(svd_eff)
        else:
            print(f"Warning: Could not parse all efficiency values from {summary_file}")
    else:
        print(f"Warning: {summary_file} not found")

# Create the plot
plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(data['ratio'], data['original'], marker='o', linewidth=2, markersize=8, 
         label='Original', color='#2ecc71')
plt.plot(data['ratio'], data['rsvd'], marker='s', linewidth=2, markersize=8, 
         label='RSVD', color='#e74c3c')
plt.plot(data['ratio'], data['svd'], marker='^', linewidth=2, markersize=8, 
         label='SVD', color='#3498db')

# Customize plot
plt.xlabel('Compression Ratio', fontsize=12, fontweight='bold')
plt.ylabel('Throughput (tokens/sec, higher is better)', fontsize=12, fontweight='bold')
plt.title('Model Efficiency vs Compression Ratio', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)

# Set x-axis ticks
plt.xticks(ratios)

plt.tight_layout()

# Save the plot
output_file = 'efficiency_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Graph saved as {output_file}")

plt.show()
