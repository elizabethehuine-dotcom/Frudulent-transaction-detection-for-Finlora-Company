from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import mlflow
import mlflow.sklearn
import pandas as pd


from src.inference.prediction import (load_historical_data,
                                      predict_transaction,
                                      engineer_feature,
                                      add_transaction_to_cache,
                                      calculate_amount_usd)

from src.utility.model_loader import load_registered_model

app = FastAPI(title='Finlora Fraud Detection API')

model = None
historical_data = None

CSV_PATH = (
    "/Users/Dell/Frudulent-transaction-detection-for-Finlora-Company/" 
    "Finlora_Dataset/Artifacts/" 
    "cleaned_customer_transaction_data.csv" )

@app.on_event("startup") 
async def startup_event(): 
    global model 
    global historical_data 
    try: 
        model = load_registered_model() 
        print( "Model loaded successfully." 
    ) 

    except Exception as e: 
        print( "Error occurred while loading " 
              f"the model: {e}" 
        ) 

        model = None 

        try: 
            historical_data = load_historical_data(
                CSV_PATH 
            ) 
            print( 
                "Historical data loaded successfully."
            )

        except Exception as e:
            print( "Error occurred while loading " 
                  f"historical dataset: {e}" 
            ) 

            historical_data = None

# what the API is expecting from the user as an input
class TransactionData(BaseModel):
    customer_id: str
    timestamp: str
    home_country: str
    source_currency: str
    dest_currency: str
    channel: str
    amount_src: float 
    fee: float
    new_device: Optional[str] = "No"
    ip_country: str
    location_mismatch: Optional[str] = "No"
    ip_risk_score: float 
    kyc_tier: str
    account_age_days: int 
    device_trust_score: float 
    risk_score_internal: float
    corridor_risk: float
       
# what the API is expecting from the user as an output or response
class PredictionResponse(BaseModel):
    is_fraud: int
    fraud_probability: float
    tnx_velocity_1h: Optional[int] = None
    tnx_velocity_24h: Optional[int] = None
    velocity_spike: Optional[int] = None
    amount_usd: Optional[float] = None

# creating the predict API so that our model can get users request and make prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionData):
    global model, historical_data

    if model is None:
        raise HTTPException(status_code=500, detail= "model nit loaded")

    if historical_data is None:
        raise HTTPException(status_code= 500, details = "historical data has not been loaded")
    
    try:
        input_data = transaction.model_dump()

# Calculate USD amount
        amount_usd = calculate_amount_usd(
            transaction.amount_src, 
            transaction.source_currency
        )

        input_data["amount_usd"] = amount_usd

# make prediction
        prediction, prediction_probability = predict_transaction(model, input_data)

        # get the engineered feature for the response
        df = pd.DataFrame([input_data])
        df_engineered = engineer_feature(df)

# Get velocity features
        tnx_velocity_1h = None
        tnx_velocity_24h = None 
        velocity_spike = None

        if "tnx_velocity_1h" in df_engineered.columns: 
            tnx_velocity_1h = int(
                df_engineered.iloc[0][
                    "tnx_velocity_1h" ])

        if "tnx_velocity_24h" in df_engineered.columns: 
            tnx_velocity_24h = int(
                df_engineered.iloc[0][
                    "tnx_velocity_24h" ])

        if "velocity_spike" in df_engineered.columns: 
            velocity_spike = int(
                df_engineered.iloc[0][
                    "velocity_spike" ]) 

        add_transaction_to_cache(
            transaction.customer_id,
            pd.to_datetime(transaction.timestamp),
            transaction.amount_src,
            amount_usd
            )

        return PredictionResponse(
            is_fraud = int(prediction),
            fraud_probability = float(prediction_probability),
            tnx_velocity_1h=(tnx_velocity_1h),
            tnx_velocity_24h=(tnx_velocity_24h),
            velocity_spike=(velocity_spike),
            amount_usd=float( amount_usd ))

    except Exception as e:
        raise HTTPException(
            status_code = 400, 
            detail = str(e))

    @app.get("/health")
    async def health_check():
        return{
            "status": "health",
            "model_loaded": model is not None,
            "historical_data_loaded": historical_data is not None
        }



    

