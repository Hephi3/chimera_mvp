import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

# Import the dataset iterator to get label mappings
from dataset.dataset_iterator import root_iter, id_to_filename

class SimplePieChartAnalyzer:
    """Simple analyzer that creates pie charts for label distributions."""
    
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
        """Load all split CSV files and organize by client and fold."""
        print("Loading split data...")
        self.split_data = {}
        
        csv_files = list(self.splits_dir.glob("splits_*.csv"))
        
        for csv_file in csv_files:
            parts = csv_file.stem.split('_')
            if len(parts) >= 3:
                client_id = int(parts[1])
                fold_id = int(parts[2])
                
                df = pd.read_csv(csv_file, index_col=0)
                
                if client_id not in self.split_data:
                    self.split_data[client_id] = {}
                
                self.split_data[client_id][fold_id] = df
        
        self.num_clients = len(self.split_data)
        self.num_folds = len(self.split_data[0]) if self.split_data else 0
        
        print(f"Loaded splits for {self.num_clients} clients and {self.num_folds} folds")
    
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
    
    def plot_client_totals(self, save_path=None):
        """Plot pie charts showing total distribution across all folds for each client."""
        fig, axes = plt.subplots(1, self.num_clients, figsize=(6 * self.num_clients, 6))
        if self.num_clients == 1:
            axes = [axes]
        
        fig.suptitle('Label Distribution per Client (All Folds Combined)', fontsize=16, fontweight='bold')
        
        for client_id in range(self.num_clients):
            ax = axes[client_id]
            
            # Combine all folds for this client
            total_train_counts = {"BRS1": 0, "BRS2": 0, "BRS3": 0}
            total_val_counts = {"BRS1": 0, "BRS2": 0, "BRS3": 0}
            total_test_counts = {"BRS1": 0, "BRS2": 0, "BRS3": 0}
            
            for fold_id in range(self.num_folds):
                if fold_id in self.split_data[client_id]:
                    df = self.split_data[client_id][fold_id]
                    
                    # Train data
                    if 'train' in df.columns:
                        train_ids = df['train'].dropna().tolist()
                        train_counts = self._get_label_counts(train_ids)
                        for label in train_counts:
                            total_train_counts[label] += train_counts[label]
                    
                    # Val data
                    if 'val' in df.columns:
                        val_ids = df['val'].dropna().tolist()
                        val_counts = self._get_label_counts(val_ids)
                        for label in val_counts:
                            total_val_counts[label] += val_counts[label]
                    
                    # Test data
                    if 'test' in df.columns:
                        test_ids = df['test'].dropna().tolist()
                        test_counts = self._get_label_counts(test_ids)
                        for label in test_counts:
                            total_test_counts[label] += test_counts[label]
            
            # Create pie chart for training data (main focus)
            labels = ['BRS1', 'BRS2', 'BRS3']
            sizes = [total_train_counts[label] for label in labels]
            total_samples = sum(sizes)
            
            if total_samples > 0:
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=self.colors, 
                                                 autopct='%1.1f%%', startangle=90)
                
                # Make percentage text bold and larger
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(12)
                
                ax.set_title(f'Client {client_id}\nTraining Data Distribution', 
                           fontweight='bold', fontsize=14)
                
                # Add detailed counts as text below the pie chart
                info_text = f"Training: BRS1={total_train_counts['BRS1']}, BRS2={total_train_counts['BRS2']}, BRS3={total_train_counts['BRS3']}\n"
                info_text += f"Validation: BRS1={total_val_counts['BRS1']}, BRS2={total_val_counts['BRS2']}, BRS3={total_val_counts['BRS3']}\n"
                info_text += f"Test: BRS1={total_test_counts['BRS1']}, BRS2={total_test_counts['BRS2']}, BRS3={total_test_counts['BRS3']}"
                
                ax.text(0, -1.5, info_text, ha='center', va='top', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_client_folds_detailed(self, save_path=None):
        """Plot pie charts for each fold of each client."""
        fig, axes = plt.subplots(self.num_clients, self.num_folds, 
                               figsize=(4 * self.num_folds, 4 * self.num_clients))
        
        if self.num_clients == 1:
            axes = axes.reshape(1, -1)
        if self.num_folds == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Label Distribution per Client and Fold (Training Data)', 
                    fontsize=16, fontweight='bold')
        
        for client_id in range(self.num_clients):
            for fold_id in range(self.num_folds):
                ax = axes[client_id, fold_id]
                
                if fold_id in self.split_data[client_id]:
                    df = self.split_data[client_id][fold_id]
                    
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
                            autotext.set_fontsize(10)
                        
                        ax.set_title(f'Client {client_id}, Fold {fold_id}', fontweight='bold')
                        
                        # Add detailed counts as text below
                        info_text = f"Train: {train_counts['BRS1']}/{train_counts['BRS2']}/{train_counts['BRS3']}\n"
                        info_text += f"Val: {val_counts['BRS1']}/{val_counts['BRS2']}/{val_counts['BRS3']}\n"
                        info_text += f"Test: {test_counts['BRS1']}/{test_counts['BRS2']}/{test_counts['BRS3']}"
                        
                        ax.text(0, -1.3, info_text, ha='center', va='top', fontsize=8,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.8))
                    else:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                               transform=ax.transAxes, fontsize=12)
                        ax.set_title(f'Client {client_id}, Fold {fold_id}')
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'Client {client_id}, Fold {fold_id}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_simple_visualizations(self, output_dir="simple_analysis"):
        """Create both visualization plots."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Creating simple pie chart visualizations...")
        
        # Plot 1: Client totals across all folds
        print("1. Creating client totals plot...")
        self.plot_client_totals(save_path=output_path / "client_totals_pie_charts.png")
        
        # Plot 2: Detailed view per fold
        print("2. Creating detailed fold-by-fold plot...")
        self.plot_client_folds_detailed(save_path=output_path / "client_folds_detailed_pie_charts.png")
        
        print(f"\nVisualizations saved to: {output_path}")
        print("Files created:")
        print("- client_totals_pie_charts.png")
        print("- client_folds_detailed_pie_charts.png")


def main():
    """Main function to run the simple analysis."""
    # splits_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cdcf/splits/chimera_3_5_0.1_1_unbalanced_0.2_0.3_0.5"
    splits_dir = "/gris/gris-f/homelv/phempel/masterthesis/MM_flower/train_cfcd/splits/chimera_3_5_2_0.1_1_unbalanced_0.8_0.2_nocd"
    
    if not os.path.exists(splits_dir):
        print(f"Error: Splits directory not found: {splits_dir}")
        return
    
    analyzer = SimplePieChartAnalyzer(splits_dir)
    analyzer.create_simple_visualizations()


if __name__ == "__main__":
    main()