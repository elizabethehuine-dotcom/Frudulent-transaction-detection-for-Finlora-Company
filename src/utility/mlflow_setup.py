import dagshub 
import mlflow 
import mlflow.sklearn

def setup_mlflow():
    """
    Set up MLflow tracking and experiment for the project.
    """

dagshub.init(repo_owner='elizabethehuine', 
             repo_name='Frudulent-transaction-detection-for-Finlora-Company', 
             mlflow=True)

# Set the MLflow experiment
mlflow.set_experiment("Fraudulent_Transaction_Detection_Model_for_Finlora_Company")