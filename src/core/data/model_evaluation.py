import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()
import joblib
# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import dagshub



# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("practice_dagshub_token")
if not dagshub_token:
    raise EnvironmentError("practice_dagshub_token environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "arpitmishra917"
repo_name = "disease-risk-app"

# Set up MLflow tracking URI
dagshub.init(repo_owner='arpitmishra917', repo_name='disease-risk-app', mlflow=True)
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/arpitmishra917/disease-risk-app.mlflow')
# dagshub.init(repo_owner='arpitmishra917', repo_name='disease-risk-app', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model(file_path: str):
    
    try:
        with open(file_path, 'rb') as file:
            model = joblib.load(file)
        logger.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(feature_path: str,target_path: str) -> pd.DataFrame:

    try:
        X= pd.read_csv(feature_path)
        y=pd.read_csv(target_path).squeeze() # Convert DataFrame to Series
        logger.info('features data loaded from %s', feature_path)
        logger.info('target data loaded from %s', target_path)
        return X,y
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def evaluate_regression_model(model,X,y,thresh = 0.485):

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        
        y_pred_thresh = (y_proba >= thresh).astype(int)

        
        Accuracy=accuracy_score(y_test, y_pred_thresh)
        Precision= precision_score(y_test, y_pred_thresh)
        Recall= recall_score(y_test, y_pred_thresh)
        F1_Score= f1_score(y_test, y_pred_thresh)
        ROC_AUC= roc_auc_score(y_test, y_proba) if y_proba is not None else None
        
        dict={
            'Accuracy': round(Accuracy, 4),
            'Precision': round(Precision, 4),
            'Recall': round(Recall, 4),
            'F1_Score': round(F1_Score, 4),
            'ROC_AUC': round(ROC_AUC, 4)
        }
        logger.info("metrices generated")
        return dict
    
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {}

def save_metrics(metrics: dict, output_dir: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "metrics.json")
        with open(path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info('Metrics saved to %s', path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, output_dir: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "model_info.json")
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.info('Model info saved to %s', path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("my-dvc-pipeline 1")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            clf = load_model('./models/logistic_regression.pkl')
            X,y = load_data(feature_path="./output/feature_engineered_data/features.csv",target_path="./output/feature_engineered_data/target.csv")

            metrics = evaluate_regression_model(clf,X,y)
            
            save_metrics(metrics, './reports')
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(sk_model=clf,name="logistic_regression_model",registered_model_name="logistic_regression_model")
            # joblib.dump(clf, "reports/logistic_regression.pkl")
            # mlflow.log_artifact("reports/logistic_regression.pkl", artifact_path="model_files")

            # Save model info
            save_model_info(run.info.run_id, "./models/logistic_regression.pkl", './reports')
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('reports/model_info.json')

        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
