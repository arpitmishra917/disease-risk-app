import os
import pandas as pd
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
FEATURE_PATH = "./output/feature_engineered_data/features.csv"
TARGET_PATH = "./output/feature_engineered_data/target.csv"
MODEL_DIR = "./models"
MODEL_NAME = "logistic_regression.pkl"

def load_data(feature_path: str, target_path: str) -> tuple[pd.DataFrame, pd.Series]:

    try:
        logger.info("Loading feature and target data")
        X = pd.read_csv(feature_path)
        y = pd.read_csv(target_path).squeeze()  # Convert DataFrame to Series
        logger.info(f"Feature shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.Series()


def data_balancing(X, y):

    try:
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logging.info(f"SMOTE applied: original={len(y)}, resampled={len(y_resampled)}")
        return X_resampled, y_resampled
    except Exception as e:
        logging.error(f"SMOTE failed: {e}")
        raise



def grid_search_cv(X_train, y_train,
                              scoring='recall', cv=5, n_jobs=-1, verbose=1,
                              random_state=42):
    try:
        model = LogisticRegression(solver='liblinear', random_state=random_state)

        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced'],
            'max_iter': [100, 200,500]
        }

        grid_search = GridSearchCV(estimator=model,
                                   param_grid=param_grid,
                                   scoring=scoring,
                                   cv=cv,
                                   n_jobs=n_jobs,
                                   verbose=verbose)

        grid_search.fit(X_train, y_train)
        logging.info(f"Best parameters: {grid_search.best_params_}")
        logging.info(f"Best score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

    except Exception as e:
        logging.error(f"GridSearchCV failed: {e}")
        raise

    
def save_model(model, output_dir: str, filename: str):

    try:
        logger.info("Saving model to disk")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")

    except Exception as e:
        logger.error("Failed to save model: %s", e)
        print(f"Error: {e}")


def main():

    try:
        logger.info("Model building pipeline started")

        # Step 1: Load data
        X, y = load_data(FEATURE_PATH, TARGET_PATH)
        if X.empty or y.empty:
            raise ValueError("Feature or target data is empty.")

        # Step 2: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # step 3: data balancing
        X_resampled, y_resampled=data_balancing(X_train,y_train)

        # Step 3: apply grid search and cross validation 
        model= grid_search_cv(X_resampled, y_resampled)

        # Step 4: Save model
        save_model(model, output_dir=MODEL_DIR, filename=MODEL_NAME)

        logger.info("Model building pipeline completed successfully")

    except Exception as e:
        logger.error("Model building pipeline failed: %s", e)
        print(f"Pipeline error: {e}")

if __name__ == '__main__':
    main()