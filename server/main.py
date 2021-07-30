import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

import pycaret.classification as pyc

from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

app = FastAPI()

class Model:
    def __init__(self, modelname, bucketname):

        self.model = pyc.load_model(modelname, platform = "aws", authentication = {'bucket':bucketname})

    def predict(self, data):

        predictions = pyc.predict_model(self.model, data=data).Label.to_list()
        return predictions



et_deployed = Model("final_et_deployed", "experimentspycaret")

rf_deployed = Model("final_rf_deployed", "experimentspycaret")

@app.post('/et/predict')

async def create_upload_file(file: UploadFile = File(...)):

    if file.filename.endswith(".csv"):

        with open(file.filename, "wb") as f:
            f.write(file.file.read())

        df = pd.read_csv(file.filename)
        os.remove(file.filename)
        try:
            return {
                "Labels" : et_deployed.predict(df)
            }

        except:
            raise HTTPException(status_code=400, detail="Invalid CSV file with column names different from the ones expected.")

    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Only csv files are accepted.")




@app.post('/rf/predict')

async def create_upload_file(file: UploadFile = File(...)):

    if file.filename.endswith(".csv"):

        with open(file.filename, "wb") as f:
            f.write(file.file.read())

        df = pd.read_csv(file.filename)
        os.remove(file.filename)
        try:
            return {
                "Labels" : rf_deployed.predict(df)
            }

        except:
            raise HTTPException(status_code=400, detail="Invalid CSV file with column names different from the ones expected.")

    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Only csv files are accepted.")



if os.getenv("AWS_ACCESS_KEY_ID") == None or os.getenv("AWS_SECRET_ACCESS_KEY") == None:
    print("AWS Credentials missing. Please set required environment variables.")