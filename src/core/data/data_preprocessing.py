import pandas as pd
import os
import logging
from scipy.stats import zscore

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ✅ Function: Import Data
def data_import(DATA_PATH):
    try:
        if not os.path.exists(DATA_PATH):
            logger.error("Raw data file not found: %s", DATA_PATH)
            print(f"Error: Raw data file not found: {DATA_PATH}")

        df_raw = pd.read_csv(DATA_PATH)
        logger.info("Raw data loaded from %s, shape=%s", DATA_PATH, df_raw.shape)
        return df_raw
    except Exception as e:
        logger.exception(f"Failed to import data from {DATA_PATH}: {e}")

# ✅ Function: Analyze and Fill Blanks
def analyze_and_fill_blanks(df):
    try:
        df_filled = df.copy()
        total_rows = len(df)

        # Calculate blank percentages (NaN + empty strings)
        blank_counts = df.isna().sum() + (df.astype(str).apply(lambda x: x.str.strip()) == '').sum()
        blank_percentage = (blank_counts / total_rows) * 100
        print("below data is percentage of blank column")
        print(blank_percentage)

        # Fill numeric columns with mean
        num_cols = df_filled.select_dtypes(include='number').columns
        for col in num_cols:
            mean_val = df_filled[col].mean()
            df_filled[col] = df_filled[col].fillna(mean_val)
            logger.info(f"Filled numeric column '{col}' with mean: {mean_val:.2f}")

        # Fill categorical columns with mode
        cat_cols = df_filled.select_dtypes(include='object').columns
        for col in cat_cols:
            mode_val = df_filled[col].mode()[0]
            df_filled[col] = df_filled[col].fillna(mode_val)
            logger.info(f"Filled categorical column '{col}' with mode: {mode_val}")

        return df_filled
    except Exception as e:
        logger.exception(f"Error during blank analysis and filling: {e}")

# ✅ Function: Remove Outliers
def remove_outliers_zscore(df, threshold=3):
    try:
        numeric_cols = df.select_dtypes(exclude='object').columns
        z_scores = df[numeric_cols].apply(zscore)

        mask = (z_scores.abs() < threshold).all(axis=1)
        df_filtered = df[mask]

        removed = len(df) - len(df_filtered)
        logger.info(f"Z-score threshold: {threshold}")
        logger.info(f"Rows before: {len(df)}, Rows after: {len(df_filtered)}, Removed: {removed}")

        return df_filtered
    except Exception as e:
        logger.exception(f"Error during outlier removal: {e}")

def save_dataframe(df, output_path):
    try:
        processed_data_path = os.path.join(output_path, 'processed_data')
        os.makedirs(processed_data_path, exist_ok=True)
        df.to_csv(os.path.join(processed_data_path,"processed_data.csv"), index=False)
        logger.info(f"Final cleaned data saved to: {output_path}")
    except Exception as e:
        logger.exception(f"Failed to save DataFrame to {output_path}: {e}")


# ✅ Main Function
def main(DATA_PATH,OUTPUT_PATH):
    try:
        logger.info("Starting data preprocessing pipeline...")

        # Step 1: Import data
        df_raw = data_import(DATA_PATH)
        if df_raw is None:
            logger.error("Pipeline terminated due to missing data.")

        # Step 2: Analyze and fill blanks
        df_filled = analyze_and_fill_blanks(df_raw)
        logger.info("Missing values handled.")

        # Step 3: Remove duplicates
        df_no_duplicates = df_filled.drop_duplicates()
        logger.info("Shape after duplicate removal: %s", df_no_duplicates.shape)

        # Step 4: Remove outliers
        df_no_outliers = remove_outliers_zscore(df_no_duplicates, threshold=3)
        logger.info("Outlier removal complete.")

        # Step 5: Save final output
        save_dataframe(df_no_outliers, OUTPUT_PATH)

        logger.info(f"Final dataset shape: {df_no_outliers.shape}")
        print("Preprocessing complete.")
        
        return None

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
    
if __name__ == "__main__":
    DATA_PATH="./output/raw/s3_data.csv"
    OUTPUT_PATH="./output"
    df_final = main(DATA_PATH, OUTPUT_PATH)