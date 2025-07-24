from fastapi import FastAPI
from app.api.quant import router as quant_router

app = FastAPI()

app.include_router(quant_router)

@app.get("/")
def read_root():
    return {"message": "Quant API is running"}

