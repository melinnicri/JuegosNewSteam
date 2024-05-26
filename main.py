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
data_gen_final = pd.read_parquet("data_gen_final.parquet")
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
@app.get("/infodeveloper/({desarrollador_normalizado})")
def InfoDeveloper(desarrollador_normalizado : str):
    '''Al ingresar el nombre de un Desarrollador, la función devuelve el Año, la cantidad de items Free y el porcentaje de éste.'''

    # Filtramos por el desarrollador que se ingrese
    filter_data_dev = data_dev[data_dev["developer"].str.lower() == desarrollador_normalizado]
    # Cantidad de items por año
    items_year = data_dev.groupby("year")["items_count"].count().to_dict()
    # Calculamos la cantidad de contenido free por año
    cantidad_free = data_dev[data_dev["price"] == 0.0].groupby("year")["items_count"].count().fillna(0).to_dict()
    # Se calcula el porcentaje free, redondeado a un decimal
    porcentaje_free = {year: f"{(cantidad_free.get(year, 0) / items_year.get(year, 1)) * 100:.1f}%" for year in items_year}
    
    # Formato de salida en JSON
    output = {
            "Agnos": items_year,
            "Cantidad de items free": cantidad_free,
            "Porcentaje free por agno": porcentaje_free
            }
    return output


# Segunda consulta:
@app.get("/userdata/({user_id})")
def UserData(User_id: str) -> dict:
    ''' Se ingresa el Usuario por id y la función retorna el Cantidad de dinero gastado, 
    Porcentaje de recomendación y Cantidad de Items '''
    
    df_user_specific = data_user[data_user["user_id"] == User_id]

    # Ver usuario
    if df_user_specific.empty:
        return {
            "Cantidad dinero gastado": None,
            "Porcentaje de Recomendación": None,
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

    output_m = {
        "Usuario": User_id,
        "Cantidad dinero gastado en USD": total_spent,
        "Porc de Recomendación": f"{recommend_percentage:.1f}",
        "Cantidad de items": items_count
    }

    return output_m


# Tercera Consulta:
@app.get("/userforgenre/({genero})")
def UserForGenre(genero: str):

    ''' Se ingresa algún Género y la función devuelve el Usuario con más horas de juego acumuladas 
        y la lista de horas de juego acumuladas por Año de Lanzamiento'''
    
    # Género si se introduce con minúscula, que funcione igual:
    genre_data = data_gen_final[data_gen_final["genres"].str.lower() == genero]

    # Verificar si el género está presente en el DataFrame
    if genero not in data_gen_final["genres"].str.lower().unique():
        return {"Error": f"El género '{genero}' no se encuentra en el conjunto de datos."}

    # Calcular la suma total de horas jugadas por usuario y género
    user_genre_playtime = data_gen_final.groupby(["user_id", "genres"])["playtime_forever"].sum().reset_index()

  #1   # Encontrar el usuario que acumula más horas jugadas por género
    max_playtime_per_genre = user_genre_playtime.loc[user_genre_playtime.groupby("genres")["playtime_forever"].idxmax()]

  #2  # Calcular la suma total de horas jugadas por año y por usuario
    playtime_per_year = data_gen_final.groupby(["user_id", "year_only"])["playtime_forever"].sum().reset_index()


    # Crear una lista de diccionarios con años y horas jugadas para cada año
    horas_por_agno = []

    available_years = playtime_per_year.loc[playtime_per_year["user_id"] == max_playtime_per_genre, "year_only"].unique()

    for year in available_years:
        horas_año = playtime_per_year[(playtime_per_year["user_id"] == max_playtime_per_genre) & (playtime_per_year["year_only"] == year)]["playtime_forever"].sum()
        if horas_año > 0:
            horas_por_agno.append({'Año': int(year), 'Horas': int(horas_año)})

        
    # Construir el diccionario final
    user_gnr_result = {
        f"Usuario con más horas jugadas para Género {genero}": max_playtime_per_genre,
        "Horas jugadas": horas_por_agno
    }
    return user_gnr_result




# Cuarta Consulta:




# Quinta Consulta:





















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