# register model

import json
import mlflow
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import dagshub
from dotenv import load_dotenv
load_dotenv()


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


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}/logistic_regression_model"
        # client = mlflow.tracking.MlflowClient()
    

        # # Replace <run_id> and <artifact_path> with your specific details
        # mlflow.register_model(
        #     model_uri=model_uri, 
        #     name="logistic_regression_model")

        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri, 
            name="logistic_regression_model" # or use the model_name variable
        )

        # # Get current recall from metrics.json
        # with open("reports/metrics.json", "r") as f:
        #     current_metrics = json.load(f)
        # current_recall = current_metrics.get("Recall", float("inf"))

        # # Compare with best Production version
        # best_recall = float("inf")
        # versions = client.search_model_versions(f"name='{model_name}'")
        # for v in versions:
        #     if v.current_stage == "Production":
        #         try:
        #             best_recall = float(v.metrics.get("Recall", float("inf")))
        #         except:
        #             pass

        # # Decide stage
        # if current_recall > best_recall:
        #     stage = "Production"
        #     archive = True
        # else:
        #     stage = "Staging"
        #     archive = False

        # # Transition stage
        # client.transition_model_version_stage(
        #     name=model_name,
        #     version=model_version.version,
        #     stage=stage,
        #     archive_existing_versions=archive
        # )

        # logging.info(f"Model {model_name} v{model_version.version} promoted to {stage} (RMSE: {current_rmse})")

    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = './reports/model_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

