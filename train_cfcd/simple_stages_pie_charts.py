import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

# Import the dataset iterator to get label mappings
from dataset.dataset_iterator import root_iter, id_to_filename

class SimpleStagesPieChartAnalyzer:
    """Simple analyzer that creates pie charts for label distributions across multiple stages."""
    
    def __init__(self, splits_dir: str):
        self.splits_dir = Path(splits_dir)
        self.label_to_int = {"BRS1": 0, "BRS2": 1, "BRS3": 2}
        self.int_to_label = {0: "BRS1", 1: "BRS2", 2: "BRS3"}
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        # Load actual labels from the dataset
        self._load_labels()
        # Load split data
        self._load_split_data()
    
    def _load_labels(self):
        """Load the actual labels for all samples from the dataset."""
        print("Loading labels from dataset...")
        self.sample_labels = {}
        
        root_iterator = root_iter(clinical_only=True)
        
        for id, data in root_iterator:
            cd_path = data["cd"]
            if os.path.exists(cd_path):
                with open(cd_path, "r") as f:
                    cd = json.load(f)
                    filename = id_to_filename(id, he=True)
                    self.sample_labels[filename] = self.label_to_int[cd["BRS"]]
        
        print(f"Loaded labels for {len(self.sample_labels)} samples")
    
    def _load_split_data(self):
        """Load all split CSV files and organize by client, fold, and stage."""
        print("Loading split data...")
        self.split_data = {}
        
        csv_files = list(self.splits_dir.glob("splits_*.csv"))
        
        for csv_file in csv_files:
            # Parse filename: splits_{client}_{fold}_{stage}.csv
            parts = csv_file.stem.split('_')
            if len(parts) >= 4:
                client_id = int(parts[1])
                fold_id = int(parts[2])
                stage_id = int(parts[3])
                
                df = pd.read_csv(csv_file, index_col=0)
                
                if stage_id not in self.split_data:
                    self.split_data[stage_id] = {}
                if client_id not in self.split_data[stage_id]:
                    self.split_data[stage_id][client_id] = {}
                
                self.split_data[stage_id][client_id][fold_id] = df
        
        self.num_stages = len(self.split_data)
        self.num_clients = len(self.split_data[0]) if self.split_data else 0
        self.num_folds = len(self.split_data[0][0]) if self.split_data and self.split_data[0] else 0
        
        print(f"Loaded splits for {self.num_stages} stages, {self.num_clients} clients and {self.num_folds} folds")
    
    def _get_label_counts(self, sample_ids):
        """Get label counts for a list of sample IDs."""
        counts = {"BRS1": 0, "BRS2": 0, "BRS3": 0}
        
        for sample_id in sample_ids:
            if pd.notna(sample_id) and sample_id.strip():
                if sample_id in self.sample_labels:
                    label_int = self.sample_labels[sample_id]
                    label_str = self.int_to_label[label_int]
                    counts[label_str] += 1
        
        return counts
    
    def plot_stages_client_totals(self, save_path=None):
        """Plot pie charts showing train+val pool distribution for each client and stage (using fold 0 as representative)."""
        fig, axes = plt.subplots(self.num_stages, self.num_clients, 
                               figsize=(6 * self.num_clients, 6 * self.num_stages))
        
        if self.num_stages == 1:
            axes = axes.reshape(1, -1)
        if self.num_clients == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Train+Val Pool Distribution per Stage and Client', 
                    fontsize=16, fontweight='bold')
        
        for stage_id in range(self.num_stages):
            for client_id in range(self.num_clients):
                ax = axes[stage_id, client_id]
                
                # Use fold 0 to get the train+val pool for this stage and client
                train_val_pool_counts = {"BRS1": 0, "BRS2": 0, "BRS3": 0}
                test_counts = {"BRS1": 0, "BRS2": 0, "BRS3": 0}
                
                if (stage_id in self.split_data and 
                    client_id in self.split_data[stage_id] and
                    0 in self.split_data[stage_id][client_id]):  # Use fold 0 as representative
                    
                    df = self.split_data[stage_id][client_id][0]
                    
                    # Combine train + val to get total available pool
                    if 'train' in df.columns:
                        train_ids = df['train'].dropna().tolist()
                        train_counts = self._get_label_counts(train_ids)
                        for label in train_counts:
                            train_val_pool_counts[label] += train_counts[label]
                    
                    if 'val' in df.columns:
                        val_ids = df['val'].dropna().tolist()
                        val_counts = self._get_label_counts(val_ids)
                        for label in val_counts:
                            train_val_pool_counts[label] += val_counts[label]
                    
                    # Test data is separate and consistent across folds
                    if 'test' in df.columns:
                        test_ids = df['test'].dropna().tolist()
                        test_counts = self._get_label_counts(test_ids)
                
                # Create pie chart for train+val pool
                labels = ['BRS1', 'BRS2', 'BRS3']
                sizes = [train_val_pool_counts[label] for label in labels]
                train_val_total = sum(sizes)
                test_total = sum(test_counts.values())
                total_client_samples = train_val_total + test_total
                
                if train_val_total > 0:
                    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=self.colors, 
                                                     autopct='%1.1f%%', startangle=90)
                    
                    # Make percentage text bold and larger
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(12)
                    
                    ax.set_title(f'Stage {stage_id}, Client {client_id}\nTrain+Val Pool Distribution\nTotal Samples: {total_client_samples} ({train_val_total} train+val, {test_total} test)', 
                               fontweight='bold', fontsize=12)
                    
                    # Add detailed counts as text below the pie chart
                    info_text = f"Train+Val Pool: BRS1={train_val_pool_counts['BRS1']}, BRS2={train_val_pool_counts['BRS2']}, BRS3={train_val_pool_counts['BRS3']}\n"
                    info_text += f"Test Set: BRS1={test_counts['BRS1']}, BRS2={test_counts['BRS2']}, BRS3={test_counts['BRS3']}"
                    
                    ax.text(0, -1.5, info_text, ha='center', va='top', fontsize=10, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'Stage {stage_id}, Client {client_id}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_stages_client_folds_detailed(self, save_path=None):
        """Plot pie charts for each fold of each client across all stages."""
        fig, axes = plt.subplots(self.num_stages * self.num_clients, self.num_folds, 
                               figsize=(4 * self.num_folds, 4 * self.num_stages * self.num_clients))
        
        # Handle single dimensions
        if self.num_stages * self.num_clients == 1:
            axes = axes.reshape(1, -1)
        if self.num_folds == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Label Distribution per Stage, Client and Fold (Training Data)', 
                    fontsize=16, fontweight='bold')
        
        row_idx = 0
        for stage_id in range(self.num_stages):
            for client_id in range(self.num_clients):
                for fold_id in range(self.num_folds):
                    ax = axes[row_idx, fold_id]
                    
                    if (stage_id in self.split_data and 
                        client_id in self.split_data[stage_id] and
                        fold_id in self.split_data[stage_id][client_id]):
                        
                        df = self.split_data[stage_id][client_id][fold_id]
                        
                        # Get counts for all splits
                        train_counts = {"BRS1": 0, "BRS2": 0, "BRS3": 0}
                        val_counts = {"BRS1": 0, "BRS2": 0, "BRS3": 0}
                        test_counts = {"BRS1": 0, "BRS2": 0, "BRS3": 0}
                        
                        if 'train' in df.columns:
                            train_ids = df['train'].dropna().tolist()
                            train_counts = self._get_label_counts(train_ids)
                        
                        if 'val' in df.columns:
                            val_ids = df['val'].dropna().tolist()
                            val_counts = self._get_label_counts(val_ids)
                        
                        if 'test' in df.columns:
                            test_ids = df['test'].dropna().tolist()
                            test_counts = self._get_label_counts(test_ids)
                        
                        # Create pie chart for training data
                        labels = ['BRS1', 'BRS2', 'BRS3']
                        sizes = [train_counts[label] for label in labels]
                        total_samples = sum(sizes)
                        
                        if total_samples > 0:
                            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=self.colors,
                                                             autopct='%1.1f%%', startangle=90)
                            
                            # Make percentage text bold
                            for autotext in autotexts:
                                autotext.set_color('white')
                                autotext.set_fontweight('bold')
                                autotext.set_fontsize(9)
                            
                            ax.set_title(f'S{stage_id} C{client_id} F{fold_id}', fontweight='bold', fontsize=10)
                            
                            # Add detailed counts as text below
                            info_text = f"Tr: {train_counts['BRS1']}/{train_counts['BRS2']}/{train_counts['BRS3']}\n"
                            info_text += f"Val: {val_counts['BRS1']}/{val_counts['BRS2']}/{val_counts['BRS3']}\n"
                            info_text += f"Test: {test_counts['BRS1']}/{test_counts['BRS2']}/{test_counts['BRS3']}"
                            
                            ax.text(0, -1.3, info_text, ha='center', va='top', fontsize=7,
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.8))
                        else:
                            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                                   transform=ax.transAxes, fontsize=10)
                            ax.set_title(f'S{stage_id} C{client_id} F{fold_id}', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                               transform=ax.transAxes, fontsize=10)
                        ax.set_title(f'S{stage_id} C{client_id} F{fold_id}', fontsize=10)
                
                row_idx += 1
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_stage_comparison(self, save_path=None):
        """Plot comparison between stages for each client (using fold 0 as representative)."""
        fig, axes = plt.subplots(1, self.num_clients, figsize=(6 * self.num_clients, 8))
        if self.num_clients == 1:
            axes = [axes]
        
        fig.suptitle('Stage Comparison per Client (Total Data Distribution)', 
                    fontsize=16, fontweight='bold')
        
        for client_id in range(self.num_clients):
            ax = axes[client_id]
            
            # Data for each stage
            stage_data = []
            stage_labels = []
            stage_info = []
            
            for stage_id in range(self.num_stages):
                # Use fold 0 to get the total data for this stage and client
                total_client_counts = {"BRS1": 0, "BRS2": 0, "BRS3": 0}
                test_counts = {"BRS1": 0, "BRS2": 0, "BRS3": 0}
                train_val_counts = {"BRS1": 0, "BRS2": 0, "BRS3": 0}
                
                if (stage_id in self.split_data and 
                    client_id in self.split_data[stage_id] and
                    0 in self.split_data[stage_id][client_id]):  # Use fold 0 as representative
                    
                    df = self.split_data[stage_id][client_id][0]
                    
                    # Train + Val data represents the total training/validation pool
                    if 'train' in df.columns:
                        train_ids = df['train'].dropna().tolist()
                        train_counts = self._get_label_counts(train_ids)
                        for label in train_counts:
                            train_val_counts[label] += train_counts[label]
                            total_client_counts[label] += train_counts[label]
                    
                    if 'val' in df.columns:
                        val_ids = df['val'].dropna().tolist()
                        val_counts = self._get_label_counts(val_ids)
                        for label in val_counts:
                            train_val_counts[label] += val_counts[label]
                            total_client_counts[label] += val_counts[label]
                    
                    # Test data is separate and consistent across folds
                    if 'test' in df.columns:
                        test_ids = df['test'].dropna().tolist()
                        test_counts = self._get_label_counts(test_ids)
                        for label in test_counts:
                            total_client_counts[label] += test_counts[label]
                
                # Store data for bar chart (use total client data)
                total_samples = sum(total_client_counts.values())
                if total_samples > 0:
                    stage_data.append([total_client_counts['BRS1'], total_client_counts['BRS2'], total_client_counts['BRS3']])
                    stage_labels.append(f'Stage {stage_id}')
                    
                    # Info text for this stage
                    info = f"Total: {total_client_counts['BRS1']}/{total_client_counts['BRS2']}/{total_client_counts['BRS3']}"
                    info += f"\nTrain+Val: {train_val_counts['BRS1']}/{train_val_counts['BRS2']}/{train_val_counts['BRS3']}"
                    info += f"\nTest: {test_counts['BRS1']}/{test_counts['BRS2']}/{test_counts['BRS3']}"
                    stage_info.append(info)
            
            if stage_data:
                # Create grouped bar chart
                stage_data = np.array(stage_data)
                x = np.arange(len(stage_labels))
                width = 0.25
                
                for i, label in enumerate(['BRS1', 'BRS2', 'BRS3']):
                    ax.bar(x + i * width, stage_data[:, i], width, label=label, 
                          color=self.colors[i], alpha=0.8)
                
                ax.set_title(f'Client {client_id}\nStage Comparison', fontweight='bold', fontsize=14)
                ax.set_xlabel('Stage')
                ax.set_ylabel('Total Number of Samples')
                ax.set_xticks(x + width)
                ax.set_xticklabels(stage_labels)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add info text below
                full_info = '\n\n'.join([f"{stage_labels[i]}:\n{stage_info[i]}" for i in range(len(stage_labels))])
                ax.text(0.5, -0.3, full_info, ha='center', va='top', fontsize=9,
                       transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_simple_visualizations(self, output_dir="simple_stages_analysis"):
        """Create all visualization plots for the stages setup."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Creating simple pie chart visualizations for stages setup...")
        
        # Plot 1: Stage and client totals across all folds
        print("1. Creating stage and client totals plot...")
        self.plot_stages_client_totals(save_path=output_path / "stages_client_totals_pie_charts.png")
        
        # Plot 2: Detailed view per fold across stages
        print("2. Creating detailed stage-client-fold plot...")
        self.plot_stages_client_folds_detailed(save_path=output_path / "stages_client_folds_detailed_pie_charts.png")
        
        # Plot 3: Stage comparison per client
        print("3. Creating stage comparison plot...")
        self.plot_stage_comparison(save_path=output_path / "stages_comparison_per_client.png")
        
        print(f"\nVisualizations saved to: {output_path}")
        print("Files created:")
        print("- stages_client_totals_pie_charts.png")
        print("- stages_client_folds_detailed_pie_charts.png")
        print("- stages_comparison_per_client.png")


def main():
    """Main function to run the stages analysis."""
    # Path to the stages splits directory - update this to your actual path
    # splits_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/splits/chimera_3_5_2_0.1_1_unbalanced_0.1_0.2_0.7"
    # splits_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/splits/chimera_3_5_2_0.1_1_unbalanced_0.8_0.2_nocd"
    # splits_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/splits/chimera_3_5_2_1_each_1_balanced_0.8_0.2_nocd"
    # splits_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/splits/chimera_3_5_2_1_each_1_unbalanced_0.8_0.2_nocd"
    # splits_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/splits/chimera_3_5_2_1_each_1_balanced_0.5_0.5_nocd"
    # splits_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/splits/chimera_3_5_2_same_1_each_3_unbalanced_0.8_0.2_nocd"
    # splits_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/splits/chimera_3_5_2_1_each_1_unbalanced_0.8_0.2_nocd_0.2"
    # splits_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/splits/chimera_3_5_2_0.2_1_0.5_0.5"
    # splits_dir = "/home/phempel/tmp_filerworkaround_phempel_nov_2025/train_cfcd/splits/chimera_3_5_2_0.2_0.8_0.2_1"
    splits_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/splits/chimera_3_5_2_0.2_0.7_0.3_1"
    
    if not os.path.exists(splits_dir):
        print(f"Error: Splits directory not found: {splits_dir}")
        print("Please update the splits_dir path in the script to point to your stages splits directory.")
        return
    
    analyzer = SimpleStagesPieChartAnalyzer(splits_dir)
    analyzer.create_simple_visualizations()


if __name__ == "__main__":
    main()