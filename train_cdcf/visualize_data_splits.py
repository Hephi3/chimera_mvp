"""
Data Split Visualization Script
This script creates comprehensive visualizations of data distribution across clients and folds
for federated learning scenarios, suitable for thesis documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Tuple, Dict
import json
import os
import glob
from collections import defaultdict, Counter
from dataset.dataset_iterator import root_iter


# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataSplitVisualizer:
    def __init__(self, split_path: str):
        """
        Initialize the visualizer with existing split data.
        
        Args:
            split_path: Path to the split directory containing CSV files, or just the split name
        """
        # Handle both full path and split name only
        if os.path.isabs(split_path) and os.path.exists(split_path):
            self.split_path = split_path
        else:
            # Assume it's just the split name, construct full path
            base_path = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train/splits"
            self.split_path = os.path.join(base_path, split_path)
        
        if not os.path.exists(self.split_path):
            raise ValueError(f"Split directory does not exist: {self.split_path}")
        
        # Label mappings
        self.label_to_int = {"BRS1": 0, "BRS2": 1, "BRS3": 2}
        self.int_to_label = {0: "BRS1", 1: "BRS2", 2: "BRS3"}
        self.split_names = ['Train', 'Validation', 'Test']
        
        # Load data and existing splits
        self._load_data()
        self._load_existing_splits()
        
    def _load_data(self):
        """Load the dataset and extract labels."""
        root_iterator = root_iter(clinical_only=True)
        
        self.ids = []
        self.labels = {}
        
        for id, data in root_iterator:
            self.ids.append(id)
            cd_path = data["cd"]
            
            with open(cd_path, "r") as input_file:
                cd = json.load(input_file)
                self.labels[id] = self.label_to_int[cd["BRS"]]
    
    def _load_existing_splits(self):
        """Load existing splits from CSV files and organize them for analysis."""
        csv_files = glob.glob(os.path.join(self.split_path, "splits_*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV split files found in {self.split_path}")
        
        self.splits_data = {}
        self.num_clients = 0
        self.folds = 0
        
        # Load CSV files and determine structure
        for csv_file in csv_files:
            basename = os.path.basename(csv_file)
            parts = basename.replace('.csv', '').split('_')
            
            if len(parts) >= 3:
                try:
                    client = int(parts[-2])
                    fold = int(parts[-1])
                    
                    self.num_clients = max(self.num_clients, client + 1)
                    self.folds = max(self.folds, fold + 1)
                    
                    # Load the CSV
                    df = pd.read_csv(csv_file)
                    
                    split_dict = {}
                    for split_name in ['train', 'val', 'test']:
                        if split_name in df.columns:
                            # Remove NaN values and convert to list
                            ids = df[split_name].dropna().tolist()
                            # Remove '_HE' suffix if present
                            ids = [id.replace('_HE', '') if id.endswith('_HE') else id for id in ids]
                            # Extract just the numeric part (e.g., '2A_001' -> '001', '2B_418' -> '418')
                            processed_ids = []
                            for id in ids:
                                if '_' in id:
                                    # Extract the part after the underscore
                                    numeric_part = id.split('_')[-1]
                                    processed_ids.append(numeric_part)
                                else:
                                    processed_ids.append(id)
                            split_dict[split_name] = processed_ids
                        else:
                            split_dict[split_name] = []
                    
                    self.splits_data[(client, fold)] = split_dict
                    
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse client/fold from filename: {basename}")
        
        print(f"Loaded splits for {self.num_clients} clients and {self.folds} folds from {self.split_path}")
        self._organize_split_data()
    
    def _organize_split_data(self):
        """Organize split data for easier analysis."""
        self.split_data = {
            'client_fold_split': defaultdict(dict),
            'overall_stats': defaultdict(list),
            'label_distributions': defaultdict(dict)
        }
        
        for client in range(self.num_clients):
            for fold in range(self.folds):
                if (client, fold) in self.splits_data:
                    split_info = self.splits_data[(client, fold)]
                    train_ids = split_info['train']
                    val_ids = split_info['val']
                    test_ids = split_info['test']
                    
                    # Store the splits
                    self.split_data['client_fold_split'][(client, fold)] = {
                        'train': train_ids,
                        'val': val_ids,
                        'test': test_ids
                    }
                    
                    # Calculate label distributions
                    for split_name, ids in zip(['train', 'val', 'test'], [train_ids, val_ids, test_ids]):
                        labels_in_split = []
                        for id in ids:
                            if id in self.labels:
                                labels_in_split.append(self.labels[id])
                            else:
                                print(f"Warning: ID {id} not found in labels dictionary")
                        
                        label_counts = Counter(labels_in_split)
                        
                        self.split_data['label_distributions'][(client, fold, split_name)] = label_counts
                        
                        # Store overall statistics
                        self.split_data['overall_stats']['sizes'].append({
                            'client': client,
                            'fold': fold,
                            'split': split_name,
                            'size': len(ids),
                            'BRS1': label_counts.get(0, 0),
                            'BRS2': label_counts.get(1, 0), 
                            'BRS3': label_counts.get(2, 0)
                        })
    
    def plot_overall_distribution(self, save_path: str = None):
        """Plot the overall label distribution in the dataset."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count overall labels
        all_labels = list(self.labels.values())
        label_counts = Counter(all_labels)
        
        # Bar plot
        labels = [self.int_to_label[i] for i in sorted(label_counts.keys())]
        counts = [label_counts[i] for i in sorted(label_counts.keys())]
        
        bars = ax1.bar(labels, counts, alpha=0.8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Overall Dataset Label Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('BRS Labels', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        wedges, texts, autotexts = ax2.pie(counts, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('Label Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_client_distribution(self, save_path: str = None):
        """Plot data distribution across clients for each fold."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Data Distribution Across {self.num_clients} Clients', 
                    fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot for each split type
        for split_idx, split_name in enumerate(['train', 'val', 'test']):
            # Sample sizes per client
            ax1 = axes[plot_idx]
            client_sizes = defaultdict(list)
            
            for client in range(self.num_clients):
                total_size = 0
                for fold in range(self.folds):
                    split_data = self.split_data['client_fold_split'][(client, fold)][split_name]
                    total_size += len(split_data)
                client_sizes[f'Client {client+1}'].append(total_size // self.folds)
            
            clients = list(client_sizes.keys())
            sizes = [client_sizes[client][0] for client in clients]
            
            bars = ax1.bar(clients, sizes, alpha=0.8, color=f'C{split_idx}')
            ax1.set_title(f'{split_name.capitalize()} Set Size per Client', fontweight='bold')
            ax1.set_ylabel('Average Sample Count')
            
            # Add value labels
            for bar, size in zip(bars, sizes):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{size}', ha='center', va='bottom', fontweight='bold')
            
            plot_idx += 1
            
            # Label distribution per client
            ax2 = axes[plot_idx]
            
            # Prepare data for stacked bar chart
            label_data = {label: [] for label in ['BRS1', 'BRS2', 'BRS3']}
            
            for client in range(self.num_clients):
                client_label_counts = Counter()
                for fold in range(self.folds):
                    fold_counts = self.split_data['label_distributions'][(client, fold, split_name)]
                    for label_int, count in fold_counts.items():
                        client_label_counts[label_int] += count
                
                # Average across folds
                for label_int in [0, 1, 2]:
                    label_name = self.int_to_label[label_int]
                    avg_count = client_label_counts.get(label_int, 0) / self.folds
                    label_data[label_name].append(avg_count)
            
            # Create stacked bar chart
            bottom = np.zeros(self.num_clients)
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, (label, counts) in enumerate(label_data.items()):
                ax2.bar(clients, counts, bottom=bottom, label=label, 
                       alpha=0.8, color=colors[i])
                bottom += counts
            
            ax2.set_title(f'{split_name.capitalize()} Set Label Distribution per Client', 
                         fontweight='bold')
            ax2.set_ylabel('Average Sample Count')
            ax2.legend()
            
            plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_data_split_composition(self, save_path: str = None):
        """Plot how data is divided into train/val/test per client and global test set composition."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Split Composition Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Data distribution per client (train/val/test proportions)
        ax1 = axes[0, 0]
        
        client_data = {'Train': [], 'Validation': [], 'Test': []}
        client_labels = []
        
        for client in range(self.num_clients):
            train_total = 0
            val_total = 0
            test_total = 0
            
            for fold in range(self.folds):
                if (client, fold) in self.split_data['client_fold_split']:
                    train_total += len(self.split_data['client_fold_split'][(client, fold)]['train'])
                    val_total += len(self.split_data['client_fold_split'][(client, fold)]['val']) 
                    test_total += len(self.split_data['client_fold_split'][(client, fold)]['test'])
            
            # Average across folds
            client_data['Train'].append(train_total / self.folds)
            client_data['Validation'].append(val_total / self.folds)
            client_data['Test'].append(test_total / self.folds)
            client_labels.append(f'Client {client+1}')
        
        # Create stacked bar chart
        bottom_val = np.array(client_data['Train'])
        bottom_test = bottom_val + np.array(client_data['Validation'])
        
        bars1 = ax1.bar(client_labels, client_data['Train'], label='Train', alpha=0.8, color='#1f77b4')
        bars2 = ax1.bar(client_labels, client_data['Validation'], bottom=bottom_val, label='Validation', alpha=0.8, color='#ff7f0e')
        bars3 = ax1.bar(client_labels, client_data['Test'], bottom=bottom_test, label='Test', alpha=0.8, color='#2ca02c')
        
        ax1.set_title('Data Split Distribution per Client\n(Average across folds)', fontweight='bold')
        ax1.set_ylabel('Number of Samples')
        ax1.legend()
        
        # Add percentage labels
        for i, client in enumerate(client_labels):
            total = client_data['Train'][i] + client_data['Validation'][i] + client_data['Test'][i]
            train_pct = (client_data['Train'][i] / total) * 100
            val_pct = (client_data['Validation'][i] / total) * 100
            test_pct = (client_data['Test'][i] / total) * 100
            
            # Add text labels
            ax1.text(i, client_data['Train'][i]/2, f'{train_pct:.1f}%', 
                    ha='center', va='center', fontweight='bold', color='white')
            ax1.text(i, bottom_val[i] + client_data['Validation'][i]/2, f'{val_pct:.1f}%', 
                    ha='center', va='center', fontweight='bold', color='white')
            ax1.text(i, bottom_test[i] + client_data['Test'][i]/2, f'{test_pct:.1f}%', 
                    ha='center', va='center', fontweight='bold', color='white')
        
        # Plot 2: Global test set composition from client test sets
        ax2 = axes[0, 1]
        
        # Collect all unique test IDs from all clients and folds
        global_test_sets = {}  # client -> set of test IDs across all folds
        all_test_ids = set()
        
        for client in range(self.num_clients):
            client_test_ids = set()
            for fold in range(self.folds):
                if (client, fold) in self.split_data['client_fold_split']:
                    test_ids = self.split_data['client_fold_split'][(client, fold)]['test']
                    client_test_ids.update(test_ids)
                    all_test_ids.update(test_ids)
            global_test_sets[client] = client_test_ids
        
        # Check for overlaps and create visualization data
        client_contributions = []
        overlap_data = []
        
        for client in range(self.num_clients):
            contribution = len(global_test_sets[client])
            client_contributions.append(contribution)
            
            # Check overlap with other clients
            overlaps = 0
            for other_client in range(self.num_clients):
                if client != other_client:
                    overlap = len(global_test_sets[client].intersection(global_test_sets[other_client]))
                    overlaps += overlap
            overlap_data.append(overlaps)
        
        # Create pie chart for global test set composition
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:self.num_clients]
        wedges, texts, autotexts = ax2.pie(client_contributions, 
                                          labels=[f'Client {i+1}' for i in range(self.num_clients)],
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        
        ax2.set_title('Global Test Set Composition\n(Contribution from each client)', fontweight='bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Plot 3: Summary statistics
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        # Hide the unused subplot
        axes[1, 1].axis('off')
        
        # Calculate summary statistics
        total_samples = len(self.ids)
        total_unique_test = len(all_test_ids)
        avg_train_per_client = np.mean(client_data['Train'])
        avg_val_per_client = np.mean(client_data['Validation'])  
        avg_test_per_client = np.mean(client_data['Test'])
        
        # Check if test sets are identical (proper global test)
        all_identical = True
        reference_set = global_test_sets[0]
        for client in range(1, self.num_clients):
            if global_test_sets[client] != reference_set:
                all_identical = False
                break
        
        summary_data = [
            ['Total Dataset Size', f'{total_samples}'],
            ['Global Test Set Size', f'{total_unique_test}'],
            ['Avg Train per Client', f'{avg_train_per_client:.1f}'],
            ['Avg Val per Client', f'{avg_val_per_client:.1f}'],
            ['Avg Test per Client', f'{avg_test_per_client:.1f}'],
            ['Test Sets Identical', 'Yes' if all_identical else 'No'],
            ['Test Set Strategy', 'Global' if all_identical else 'Local']
        ]
        
        table = ax3.table(cellText=summary_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.4, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    cell.set_text_props(weight='bold' if j == 0 else 'normal')
        
        ax3.set_title('Split Summary Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    

    



def main(split_name: str = "chimera_3_5_0.1_5"):
    """Main function to generate visualizations."""
    # You can provide either:
    # 1. Just the split name (e.g., "chimera_3_5_0.1_5")
    # 2. Full path to the split directory
    split_path = split_name
    
    # Create output directory
    output_dir = f"visualization_output_{split_name.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating Data Split Visualizer for: {split_path}")
    try:
        visualizer = DataSplitVisualizer(split_path=split_path)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease provide either:")
        print("1. Just the split name (e.g., 'chimera_3_5_0.1_5')")
        print("2. Full path to the split directory")
        return
    
    print("Generating visualizations...")
    
    # Generate the essential plots
    visualizer.plot_overall_distribution(
        save_path=os.path.join(output_dir, "01_overall_distribution.png")
    )
    
    visualizer.plot_client_distribution(
        save_path=os.path.join(output_dir, "02_client_distribution.png")
    )
    
    visualizer.plot_data_split_composition(
        save_path=os.path.join(output_dir, "03_data_split_composition.png")
    )
    
    print(f"\nVisualizations saved to: {output_dir}/")


if __name__ == "__main__":
    import sys
    
    # Allow command line argument for split name/path
    if len(sys.argv) > 1:
        split_name = sys.argv[1]
    else:
        split_name = "chimera_3_5_0.1_5"  # Default split name
    
    main(split_name)