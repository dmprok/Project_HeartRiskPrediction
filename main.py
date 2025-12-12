from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from joblib import load
import pandas as pd
import os
import uuid
import uvicorn
import logging
from encoder import CustomEncoder
app = FastAPI()
templates = Jinja2Templates(directory="templates")

logging.basicConfig(
    filename='log.txt',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
@app.get("/", tags=["Корневой каталог"], summary="Приветствие")
def enter(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
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
        predictions = model.predict(data_prep)
        result['prediction'] = predictions

        unique_filename = f'{uuid.uuid4().hex}.csv'
        result_path = os.path.join("predictions", unique_filename)
        result.to_csv(result_path, index=False)
        logging.info(f'Результат записан в файл: {unique_filename}')

        download_url = f'http://localhost:8000/predict/download?filename={unique_filename}'

        json_response = {
            "status": "ok",
            "download_link": download_url,
            "results": result.to_dict(orient="records"),
        }
        return JSONResponse(content=json_response)
    except Exception as e:
        logging.warning(f'file processing error: {e}')
        raise HTTPException(status_code=400, detail="File processing error")

@app.get("/predict/download", response_class=FileResponse, tags=["Предсказания"], summary="Выгрузка предсказания в формате csv")
def get_predict_csv(filename: str):
    path_to_file = os.path.join("predictions", filename)
    if not os.path.exists(path_to_file):
        raise HTTPException(status_code=404, detail=f"File {path_to_file} not found")
    return FileResponse(str(path_to_file), filename=filename)

if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)