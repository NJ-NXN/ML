from fastapi import FastAPI
from fastapi.responses import Response

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": {"My first API"}}

@app.get("/csv")
def get_csv():
    csv_content = {
        "product, price, stock\n"
        "Laptop, 999, 42\n"
        "Mouse, 29, 150"
    }
    return Response{
        content=csv_content, 
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=products.csv"}
        }