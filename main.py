# Importa librerías necesarias
from fastapi import FastAPI
import pandas as pd
import pyarrow
import fastparquet
import parquet

import sklearn as sk
from sklearn.metrics.pairwise import cosine_similarity
import operator
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer

# Usar datasets desde parquet (consultas)
data_dev = pd.read_parquet("data_dev.parquet")
data_user = pd.read_parquet("data_user.parquet")
piv_table_norm = pd.read_parquet("piv_table_norm.parquet")
df_user_simil =pd.read_parquet("df_user_simil.parquet")
cosine_sim_df = pd.read_parquet("cosine_sim_df.parquet")


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
@app.get("/developerinfo/({dev})")
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


# Segunda consulta:
@app.get("/userdata/({user_id})")
def UserData(User_id: str) -> dict:
    ''' Se ingresa el Usuario por id y la función retorna el Cantidad de dinero gastado, 
    Porcentaje de recomendación y Cantidad de Items '''
    
    df_user_specific = data_user[data_user["user_id"] == User_id]

    # Handle empty DataFrame gracefully
    if df_user_specific.empty:
        return {
            "Cantidad dinero gastado": None,
            "% de Recomendación": None,
            "Cantidad de items": None
        }

    total_dinero_por_usuario = df_user_specific.groupby("user_id")["price"].sum()
    user_grouped = (
        data_user.groupby("user_id")["recommend"]
        .apply(lambda x: (x == True).mean() * 100)
        .reset_index(name="recommend_true_porcentaje")
    )
    user_counts = (
        data_user.groupby("user_id")
        .size()
        .to_frame(name="items_count")
        .reset_index()
    )

    # Use try-except for potential index errors
    try:
        total_spent = total_dinero_por_usuario.iloc[0]
    except IndexError:
        total_spent = None

    try:
        recommend_percentage = user_grouped[user_grouped["user_id"] == User_id]["recommend_true_porcentaje"].iloc[0]
    except IndexError:
        recommend_percentage = None

    try:
        items_count = user_counts[user_counts["user_id"] == User_id]["items_count"].iloc[0]
    except IndexError:
        items_count = None

    output = {
        "Usuario": User_id,
        "Cantidad dinero gastado en USD": total_spent,
        "Porc de Recomendación": f"{recommend_percentage:.1f}",
        "Cantidad de items": items_count
    }

    return output




# ML: RECOMENDACIÓN USER-ITEM:

@app.get("/similaruserrecs/({user})")
def similar_user_recs(user):
  
    '''Los 5 juegos más recomendados similares recomendados por usuario...'''
    # Se verifica si el usuario está presente en las columnas de piv_table_norm
    if user not in df_user_simil.columns:
        return {'message': 'El Usuario no tiene datos disponibles {}'.format(user)}

    # Se obtienen los usuarios más similares 
    sim_users = df_user_simil.sort_values(by=user, ascending=False).index[1:11]

    best = []  
    most_common = {}  

    # Por cada usuario similar, encuentra el juego mejor calificado y lo agrega a la lista 'best'
    for i in sim_users:
        max_score = piv_table_norm.loc[:, i].max()
        best.append(piv_table_norm[piv_table_norm.loc[:, i] == max_score].index.tolist())

    # Se cuenta cuántas veces se recomienda cada juego
    for i in range(len(best)):
        for j in best[i]:
            if j in most_common:
                most_common[j] += 1
            else:
                most_common[j] = 1

    # Se ordenan los juegos de mayor recomendación
    sorted_list = sorted(most_common.items(), key=lambda x: x[1], reverse=True)

    return dict(sorted_list[:5])



# RECOMENDACIÓN ITEM-ITEM:

@app.get("/getsimilaritems/({item_id})")
def get_similar_items(item_id, top_n=5):
    ''' La función para obtener el top N=5 de items similares al introducido por id de juego'''

    similar_items = cosine_sim_df[item_id].sort_values(ascending=False).head(top_n + 1).iloc[1:]
    return similar_items