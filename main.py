# Importa librerías necesarias
from fastapi import FastAPI
import pandas as pd
import pyarrow
import fastparquet
import parquet

# Usar datasets desde parquet (consultas)
data_dev = pd.read_parquet("data_dev.parquet")


# Se instancia la aplicación
app = FastAPI(title="PROYECTO INDIVIDUAL Nº1 - Machine Learning Operations (MLOps) - Amelia Herrera Briceño - melinnicri@gmail.com",
            description="API de datos y recomendaciones de juegos de video online STEAM")


# Función para reconocer el servidor local
@app.get("/")
async def index():
    return {"Hola! Bienvenido a la API de consulta y recomendación. Por favor dirígete a /docs"}

@app.get("/about/")
async def about():
    return {"PROYECTO INDIVIDUAL Nº1 -Machine Learning Operations (MLOps)"}


# Primera consulta:
@app.get("/developer_info/({dev})")
def developer_info(dev: str):
    ''' Se ingresa el Desarrollador y la función retorna el Año, Cantidad de Items y Cantidad de Free en % '''

    # Filtramos por el desarrollador que se ingrese
    filter_data_dev = data_dev[data_dev["developer"] == dev]


    # Se la cantidad de items por año
    items_per_year = data_dev.groupby("year")["items_count"].count().to_dict()
    
    # Calculamos la cantidad de contenido free por año y convertimos a un diccionario
    free_count = data_dev[data_dev["price"] == 0.0].groupby("year")["items_count"].count().fillna(0).to_dict()
    
    # Calculamos el porcentaje free
    porcentaje_free = {year: (free_count.get(year, 0) / items_per_year.get(year, 1)) * 100 for year in items_per_year}
    
    developer_dict = {}
    for year in items_per_year.keys():
        year_int = int(year)
        developer_dict[year_int] = {
            "Agno": year,
            "Cantidad de Items": items_per_year.get(year, 0),
            f"% Free": round(porcentaje_free.get(year, 0), 1)
        }
    return developer_dict