import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import os

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def feature_engineering(df):
    try:
        df_engg = df.copy()

        # BMI category
        df_engg['bmi_category'] = pd.cut(
            df_engg['bmi'],
            bins=[0, 18.5, 24.9, 29.9, np.inf],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        logger.info("Created 'bmi_category' feature")

        # Blood pressure category (systolic only)
        df_engg['bp_category'] = pd.cut(
            df_engg['systolic_bp'],
            bins=[0, 120, 129, 139, 180, np.inf],
            labels=['Normal', 'Elevated', 'Stage1', 'Stage2', 'Crisis']
        )
        logger.info("Created 'bp_category' feature")

        # Activity-to-sleep ratio
        df_engg['activity_sleep_ratio'] = df_engg['daily_steps'] / (df_engg['sleep_hours'] + 1e-3)
        logger.info("Created 'activity_sleep_ratio' feature")

        # Hydration index
        df_engg['hydration_index'] = df_engg['water_intake_l'] / (df_engg['calories_consumed'] + 1e-3)
        logger.info("Created 'hydration_index' feature")

        # Interaction terms
        df_engg['age_cholesterol'] = df_engg['age'] * df_engg['cholesterol']
        df_engg['smoker_alcohol'] = df_engg['smoker'] * df_engg['alcohol']
        df_engg['resting_bp_product'] = df_engg['resting_hr'] * df_engg['systolic_bp']
        logger.info("Created interaction features: 'age_cholesterol', 'smoker_alcohol', 'resting_bp_product'")

        return df_engg

    except Exception as e:
        logger.exception(f"Feature engineering failed: {e}")


def encode_features(df, ordinal_map=None, cardinality_threshold=3):
    try:
        df_encoded = df.copy()

        for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
            unique_vals = df_encoded[col].nunique()

            # Ordinal encoding
            if ordinal_map and col in ordinal_map:
                ordered_categories = ordinal_map[col]
                df_encoded[col] = df_encoded[col].astype(
                    pd.CategoricalDtype(categories=ordered_categories, ordered=True)
                ).cat.codes
                logger.info(f"Ordinal encoded column: {col}")

            # One-hot encoding for low-cardinality nominal features
            elif unique_vals <= cardinality_threshold:
                df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
                logger.info(f"One-hot encoded column: {col}")

            # Label encoding for high-cardinality nominal features
            else:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                logger.info(f"Label encoded column: {col}")

        # Convert boolean columns to integers
        bool_cols = df_encoded.select_dtypes(include='bool').columns
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
        logger.info(f"Converted boolean columns to int: {list(bool_cols)}")

        return df_encoded

    except Exception as e:
        logger.exception(f"Feature encoding failed: {e}")


def scale_features(df, target_column):
    try:
        y = df[target_column]
        x = df.drop(columns=target_column)
        scaler=StandardScaler()

        x_scaled = pd.DataFrame(
            scaler.fit_transform(x),
            columns=x.columns,
            index=x.index
        )

        logger.info(f"Feature scaling complete using {scaler.__class__.__name__}")
        logger.info(f"Scaled shape: {x_scaled.shape}, Target shape: {y.shape}")

        return x_scaled, y

    except Exception as e:
        logger.exception(f"Feature scaling failed: {e}")


def get_top_features(X,y,top_n=10, random_state=42):

    try:
        logger.info("Getting top features")

        model = RandomForestRegressor(n_estimators=50,random_state=random_state)
        model.fit(X, y)

        importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features_series = importances.sort_values(ascending=False).head(top_n)
        top_features=top_features_series.index.tolist()
        logger.info(f"Top features: {top_features}")
        return top_features
    except Exception as e:
        logger.error('Failed in getting top features: %s', e)

def filter_top_features(df, top_features):
    try:
        logger.info("Filtering top features")
        return df[top_features].copy()
    except Exception as e:
        logger.error('Failed in extracting top features: %s', e)


def save_output(X, y, output_dir, x_filename='features.csv', y_filename='target.csv'):
   
    try:
        logger.info("Saving X and y separately to files")
        os.makedirs(output_dir, exist_ok=True)

        x_path = os.path.join(output_dir, x_filename)
        y_path = os.path.join(output_dir, y_filename)

        X.to_csv(x_path, index=False)
        y.to_frame().to_csv(y_path, index=False)  # Convert y to DataFrame before saving

        logger.info(f"Features saved to {x_path}")
        logger.info(f"Target saved to {y_path}")
    except Exception as e:
        logger.error("Failed to save X and y separately: %s", e)


def main(df_raw, target_column, output_dir):
    try:
        logger.info("Starting preprocessing pipeline...")

        # Step 1: Feature Engineering
        df_engg = feature_engineering(df_raw)

        # Step 2: Encoding
        ordinal_map = {
            "bmi_category": ["Underweight", "Normal", "Overweight", "Obese"],
            "bp_category": ["Normal", "Elevated", "Stage1", "Stage2", "Crisis"]
        }
        df_encoded = encode_features(df_engg, ordinal_map=ordinal_map)


        # Step 3: Scaling
        scaled_X, y = scale_features(df_encoded, target_column=target_column)

        # step 4: filtering top features
        top_features = get_top_features(scaled_X,y)
        df_top_features_X = filter_top_features(scaled_X, top_features)
       
        # Step 5: Save output
        save_output(df_top_features_X, y, output_dir=output_dir)

        logger.info("feature engg. pipeline completed successfully.")

    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    # Load your raw DataFrame however you prefer
    df_raw = pd.read_csv("./output/processed_data/processed_data.csv")
    target_column = "disease_risk"
    output_dir = "./output/feature_engineered_data"

    main(df_raw, target_column, output_dir)