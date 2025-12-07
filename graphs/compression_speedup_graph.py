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
    'rsvd_time': [],
    'svd_time': []
}

# Read data from compression_summary.txt files
base_dir = '../benchmark_results_comparison'

for ratio in ratios:
    summary_file = os.path.join(base_dir, f'ratio_{ratio}', 'compression_summary.txt')
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            content = f.read()
            
        # Parse compression timing values
        rsvd_time = None
        svd_time = None
        
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('1. RSVD (with whitening):'):
                rsvd_time = float(line.split(':')[1].strip().split()[0])
            elif line.startswith('2. SVD (with whitening):'):
                svd_time = float(line.split(':')[1].strip().split()[0])
        
        # Only append if all values were found
        if rsvd_time is not None and svd_time is not None:
            data['ratio'].append(ratio)
            data['rsvd_time'].append(rsvd_time)
            data['svd_time'].append(svd_time)
        else:
            print(f"Warning: Could not parse all compression timing values from {summary_file}")
    else:
        print(f"Warning: {summary_file} not found")

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(data['ratio']))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, data['rsvd_time'], width, label='RSVD', color='#e74c3c')
bars2 = ax.bar(x + width/2, data['svd_time'], width, label='SVD', color='#3498db')

# Customize plot
ax.set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('Compression Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Compression Time Comparison: RSVD vs SVD', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([1 - r for r in data['ratio']])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save the plot
output_file = 'compression_time_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Graph saved as {output_file}")

plt.show()
