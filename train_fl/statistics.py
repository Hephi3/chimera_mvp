
import numpy as np
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict

def get_patch_statistics():
    """Generate comprehensive statistics about patches per HE file across all pages."""
    
    base_path = "/local/scratch/phempel/chimera/features_1536"
    pages = ["coordinates_page0", "coordinates_page1", "coordinates_page2", 
             "coordinates_page3", "coordinates_page4"]
    
    # Dictionary to store patch counts: {filename: {page: count}}
    patch_counts = defaultdict(dict)
    page_statistics = {}
    
    print("Analyzing patch coordinates across all pages...\n")
    
    # Process each page
    for page in pages:
        page_path = os.path.join(base_path, page)
        
        if not os.path.exists(page_path):
            print(f"Warning: Path {page_path} does not exist")
            continue
            
        npy_files = [f for f in os.listdir(page_path) if f.endswith('.npy')]
        page_patch_counts = []
        
        print(f"Processing {page} ({len(npy_files)} files)...")
        
        for npy_file in npy_files:
            file_path = os.path.join(page_path, npy_file)
            try:
                coordinates = np.load(file_path)
                num_patches = len(coordinates)
                
                # Store patch count for this file and page
                filename = npy_file.replace('.npy', '')
                patch_counts[filename][page] = num_patches
                page_patch_counts.append(num_patches)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Calculate statistics for this page
        if page_patch_counts:
            page_statistics[page] = {
                'count': len(page_patch_counts),
                'mean': np.mean(page_patch_counts),
                'median': np.median(page_patch_counts),
                'std': np.std(page_patch_counts),
                'min': np.min(page_patch_counts),
                'max': np.max(page_patch_counts),
                'total_patches': np.sum(page_patch_counts)
            }
    
    return patch_counts, page_statistics

def print_statistics(patch_counts, page_statistics):
    """Print comprehensive statistics."""
    
    print("=" * 60)
    print("PATCH STATISTICS SUMMARY")
    print("=" * 60)
    
    # Per-page statistics
    print("\n📊 STATISTICS PER PAGE:")
    print("-" * 40)
    for page, stats in page_statistics.items():
        print(f"\n{page.upper()}:")
        print(f"  Files processed: {stats['count']}")
        print(f"  Total patches: {stats['total_patches']:,}")
        print(f"  Average patches per file: {stats['mean']:.1f}")
        print(f"  Median patches per file: {stats['median']:.1f}")
        print(f"  Standard deviation: {stats['std']:.1f}")
        print(f"  Min patches: {stats['min']}")
        print(f"  Max patches: {stats['max']}")
    
    # Overall statistics across all pages
    print("\n📈 OVERALL STATISTICS:")
    print("-" * 40)
    
    # Calculate overall statistics
    all_patch_counts = []
    total_patches_all = 0
    
    for filename, page_data in patch_counts.items():
        file_total = sum(page_data.values())
        all_patch_counts.append(file_total)
        total_patches_all += file_total
    
    if all_patch_counts:
        print(f"Total HE files analyzed: {len(patch_counts)}")
        print(f"Total patches across all files and pages: {total_patches_all:,}")
        print(f"Average patches per HE file (all pages): {np.mean(all_patch_counts):.1f}")
        print(f"Median patches per HE file (all pages): {np.median(all_patch_counts):.1f}")
        print(f"Standard deviation: {np.std(all_patch_counts):.1f}")
        print(f"Min patches per HE file: {np.min(all_patch_counts)}")
        print(f"Max patches per HE file: {np.max(all_patch_counts)}")
    
    # Files with data across multiple pages
    print("\n🔍 MULTI-PAGE ANALYSIS:")
    print("-" * 40)
    
    files_per_page_count = defaultdict(int)
    for filename, page_data in patch_counts.items():
        num_pages = len(page_data)
        files_per_page_count[num_pages] += 1
    
    for num_pages, count in sorted(files_per_page_count.items()):
        print(f"Files present in {num_pages} page(s): {count}")

def save_detailed_report(patch_counts, page_statistics):
    """Save detailed statistics to CSV files."""
    
    # Create DataFrame with detailed file information
    detailed_data = []
    for filename, page_data in patch_counts.items():
        row = {'filename': filename}
        total_patches = 0
        
        for page in ["coordinates_page0", "coordinates_page1", "coordinates_page2", 
                     "coordinates_page3", "coordinates_page4"]:
            patches = page_data.get(page, 0)
            row[page] = patches
            total_patches += patches
        
        row['total_patches'] = total_patches
        row['num_pages'] = len([p for p in page_data.values() if p > 0])
        detailed_data.append(row)
    
    df = pd.DataFrame(detailed_data)
    df.to_csv('patch_statistics_detailed.csv', index=False)
    print(f"\n💾 Detailed statistics saved to: patch_statistics_detailed.csv")
    
    # Create summary DataFrame
    summary_data = []
    for page, stats in page_statistics.items():
        summary_data.append({
            'page': page,
            'files_count': stats['count'],
            'total_patches': stats['total_patches'],
            'mean_patches': round(stats['mean'], 1),
            'median_patches': stats['median'],
            'std_patches': round(stats['std'], 1),
            'min_patches': stats['min'],
            'max_patches': stats['max']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('patch_statistics_summary.csv', index=False)
    print(f"💾 Summary statistics saved to: patch_statistics_summary.csv")

def main():
    """Main function to run the statistics analysis."""
    try:
        patch_counts, page_statistics = get_patch_statistics()
        print_statistics(patch_counts, page_statistics)
        save_detailed_report(patch_counts, page_statistics)
        
        print("\n✅ Statistics analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()