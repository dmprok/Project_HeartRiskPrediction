from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from joblib import load
import pandas as pd
import uvicorn
import logging
from encoder import CustomEncoder
app = FastAPI()

logging.basicConfig(
    filename='log.txt',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

@app.get("/", tags=["Корневой каталог"], summary="Приветствие")
def enter():
    return "Hello stranger! Do you want to predict?"
@app.post("/predict", tags=["Предсказания"], summary="Загрузка csv и выполнение предсказания")
def predict_risk(file: UploadFile = File()):
    model = load('best_model.joblib')
    preprocessor = load('data_preprocessor.joblib')

    try:
        data = pd.read_csv(file.file)
        logging.info(f'processing file - {file.filename}')
    except:
        logging.info(f'wrong format of processing file - {file.filename}')
        raise HTTPException(status_code=406, detail="File format not supported")

    try:
        data_prep = preprocessor.fit_transform(data)
        result = data[['id']]
        prediction = model.predict(data_prep)
        result.loc[:, 'prediction'] = prediction
        name = str(file.filename)[:-4] + '_prediction.csv'
        result.to_csv(name, index=False)
        result_json = result.to_json(orient='records')
        logging.info(f'successful processing file - {file.filename}')
        return result_json
    except:
        logging.warning(f'file processing error - {file.filename}')
        raise HTTPException(status_code=400, detail="Error")

@app.get("/predict/download", tags=["Предсказания"], summary="Выгрузка предсказания в csv")
def get_predict_csv(file_name: str):
    try:
        result_name = file_name + '_prediction.csv'
        logging.info(f'downloading {result_name}')
        return FileResponse(result_name, filename=result_name)
    except:
        logging.info(f'downloading {result_name} error')
        raise HTTPException(status_code=400, detail="File not found.")

if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)
