import pandas as pd
import os

def merge_stages_columnwise(input_csv_stage0, input_csv_stage1, output_csv):
    df0 = pd.read_csv(input_csv_stage0, index_col=False)
    df1 = pd.read_csv(input_csv_stage1, index_col=False)

    # Drop "Unnamed: 0" if present
    for df in [df0, df1]:
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    # For each column, concatenate non-empty values from both stages
    merged = {}
    for col in ['train', 'val', 'test']:
        col0 = df0[col].dropna().astype(str)
        col1 = df1[col].dropna().astype(str)
        # Remove empty strings
        col0 = col0[col0 != ""]
        col1 = col1[col1 != ""]
        merged[col] = pd.concat([col0, col1], ignore_index=True)

    # Find the maximum length
    max_len = max(len(merged['train']), len(merged['val']), len(merged['test']))

    # Pad each column with empty strings to the same length
    for col in merged:
        merged[col] = merged[col].reindex(range(max_len), fill_value="")

    # Create DataFrame and add index
    merged_df = pd.DataFrame(merged)
    merged_df.index.name = ""  # Optional: remove index name

    # Save to CSV with index
    merged_df.to_csv(output_csv, index=True)
    print(f"Merged CSV saved to {output_csv}")

if __name__ == "__main__":
    seed = 5
    source_dir = f"/home/phempel/tmp_filerworkaround_phempel_nov_2025/train_cfcd/train_cfcd/splits/chimera_3_5_2_same_1_each_balanced_0.5_0.5_nocd_merged_{seed}"
    dest_dir = f"/home/phempel/tmp_filerworkaround_phempel_nov_2025/train_cfcd/train_cfcd/splits/chimera_3_5_2_same_1_each_balanced_0.5_0.5_nocd_merged_merged_{seed}"
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # for client in range(3):
    for fold in range(5):
        input_csv_stage0 = os.path.join(source_dir, f"splits_0_{fold}_0.csv")
        input_csv_stage1 = os.path.join(source_dir, f"splits_0_{fold}_1.csv")
        output_csv = os.path.join(dest_dir, f"splits_0_{fold}.csv")
        merge_stages_columnwise(input_csv_stage0, input_csv_stage1, output_csv)