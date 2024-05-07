from fastapi import FastAPI , Path, UploadFile, File
from typing import Optional
from pydantic import BaseModel
import shutil
import pandas as pd

app=FastAPI(title='Consultas por Streaming',
description= ' En esta API se podran realizar consultas a los datos de la plataforma Steam teniendo un total de 5 funciones',
version='1.1.1.1')

steam_games = pd.read_parquet(r'C:\\Users\\licle\\Desktop\\Prep Henry\\Proyecto Integrados n1\\PI MLOps - STEAM\\Datasets\\steam_games.parquet')
items_def = pd.read_parquet(r'C:\\Users\\licle\\Desktop\\Prep Henry\\Proyecto Integrados n1\\PI MLOps - STEAM\\Datasets\\users_items.parquet')
reviews_def = pd.read_parquet(r'C:\\Users\\licle\\Desktop\\Prep Henry\\Proyecto Integrados n1\\PI MLOps - STEAM\\Datasets\\user_reviews.parquet')

@app.get('/PlayTimeGenre')
async def PlayTimeGenre(genero : str):
    merged_data = pd.merge(steam_games, items_def, left_on='id', right_on='item_id', how='inner')
    genres = merged_data.explode('genres') 
    genres['genres'] = genres['genres'].str.replace('&amp;', 'and')
    genres['genres'] = genres['genres'].str.lower().str.replace(r'[^a-zA-Z0-9\s]', '').str.strip()
    unique_genres = genres['genres'].unique()
    user_genre = genero
    user_genre = user_genre.lower().strip()
    if user_genre.lower() not in unique_genres:
        print("Error: El género no está en la lista")
    else:
        genre_data = genres[genres['genres'] == user_genre]
    minutes_played_by_year = genre_data.groupby(genre_data['release_date'].dt.year)['playtime_forever'].sum()
    most_played_year = int(minutes_played_by_year.idxmax())
    return print("Año de lanzamiento con más horas jugadas para el género", user_genre, ":", most_played_year)

@app.get('/UserForGenre')
async def UserForGenre(genero : str):
    merged_data = pd.merge(steam_games, items_def, left_on='id', right_on='item_id', how='inner')
    genres = merged_data.explode('genres')
    genres['genres'] = genres['genres'].str.replace('&amp;', '&')
    genres['genres'] = genres['genres'].str.lower().str.replace(r'[^a-zA-Z0-9\s]', '').str.strip()
    unique_genres = genres['genres'].unique()
    user_genre = genero
    user_genre = user_genre.lower().strip()
    if user_genre.lower() not in unique_genres:
        print("Error: El género no está en la lista")
    else:
        genre_data = genres[genres['genres'] == user_genre]
        most_played_user = genre_data.groupby('user_id')['playtime_forever'].sum().idxmax()
        most_played_user_data = genre_data[genre_data['user_id'] == most_played_user]
        minutes_played_by_year = most_played_user_data.groupby(most_played_user_data['release_date'].dt.year)['playtime_forever'].sum()
        hours_by_year_list = [{"Año": int(year), "Horas": hours/60} for year, hours in minutes_played_by_year.items()]
    return print("Usuario con más horas jugadas para Género", user_genre.capitalize(), ":", most_played_user, ' Horas jugadas : ', hours_by_year_list)

@app.get('/UsersRecommend')
async def UsersRecommend(año: int):
    reviews_def['item_id'] = reviews_def['item_id'].astype(int)
    steam_games['id'] = steam_games['id'].astype(int)
    merged_data = pd.merge(reviews_def, steam_games, left_on='item_id', right_on='id', how='inner')
    year_games = merged_data[merged_data['release_date'].dt.year == año]
    recommended_games = year_games[(year_games['recommend'] == True) & (year_games['sentiment_analysis'] >= 1)]
    recommend_count = recommended_games['app_name'].value_counts()
    recomendados = recommend_count.head(3)
    respuesta3 = [{"Puesto " + str(i+1): juego} for i, juego in enumerate(recomendados.index)]
    return respuesta3

@app.get('/UsersNotRecommend')
async def UsersNotRecommend(año : int):
    reviews_def['item_id'] = reviews_def['item_id'].astype(int)
    steam_games['id'] = steam_games['id'].astype(int)
    merged_data = pd.merge(reviews_def, steam_games, left_on='item_id', right_on='id', how='inner')
    year_games = merged_data[merged_data['release_date'].dt.year == año]
    least_recommended_games = year_games[(year_games['recommend'] == False) & (year_games['sentiment_analysis'] == 0)]
    least_recommend_count = least_recommended_games['app_name'].value_counts()
    menos_recomendados = least_recommend_count.head(3)
    respuesta4 = [{"Puesto " + str(i+1): juego} for i, juego in enumerate(menos_recomendados.index)]
    
    return respuesta4

@app.get('/sentiment_analysis')
async def sentiment_analysis(año: int):
    merged_data = pd.merge(reviews_def, steam_games, left_on='item_id', right_on='id', how='inner')
    filtro_resenas = merged_data[merged_data['release_date'].dt.year == año]
    sentiment_count = filtro_resenas['sentiment_analysis'].value_counts()
    respuesta5 = {
        "Negative": sentiment_count.get(0, 0),
        "Neutral": sentiment_count.get(1, 0),
        "Positive": sentiment_count.get(2, 0)
    }
    return respuesta5