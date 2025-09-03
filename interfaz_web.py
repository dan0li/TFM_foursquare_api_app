import streamlit as st
import requests
import numpy as np
import json
from geopy.geocoders import Nominatim
from geopy.geocoders import OpenCage
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon
from geopy.distance import distance as geodistance
from geopy.distance import geodesic
import pandas as pd
import plotly.express as px
import io
import geopandas as gpd
import zipfile
import tempfile
import os
import math
import ast
import re
import dask.dataframe as dd

# Configuramos las API KEYS necesarias
API_KEY = st.secrets["FOURSQUARE_API_KEY"]
hf_token = st.secrets["HF_TOKEN"]
OPENCAGE_KEY = st.secrets["OPENCAGE_KEY"]
    
# Cargar el dataset de Foursquare
len_df = st.session_state.get('len_HF_dataset', None)
if len_df is None:
    df = dd.read_parquet("hf://datasets/foursquare/fsq-os-places/release/dt=2025-08-07/places/parquet/*.parquet",
                         storage_options={"token": hf_token})
    st.session_state['len_HF_dataset'] = df.shape[0].compute()

# Cargar el dataset de categorias de Foursquare
categories_fsq = st.session_state.get('categories_FSQ', None)
if categories_fsq is None:
    categories_fsq = dd.read_parquet("hf://datasets/foursquare/fsq-os-places/release/dt=2025-08-07/categories/parquet/categories.zstd.parquet",
                                     storage_options={"token": hf_token})
    st.session_state['categories_FSQ'] = categories_fsq.compute()

st.title("Descargar Datos de Foursquare")

# Inicializar parametros de sesión
st.session_state['show_map'] = False
st.session_state['analytics'] = False


parametros = {}

location = st.text_input("Ubicación:", None)
parametros['near'] = location
query = st.text_input("Consulta:")
if query:
    parametros['query'] = query
categories = st.text_input("Categorías:")
if categories:
    parametros['categories'] = categories
limit = st.number_input("Número de resultados a mostrar:", min_value=1, value=10)
parametros['limit'] = limit

@st.cache_data(show_spinner=False)
def geocode_location(place: str):
    geolocator = OpenCage(api_key=OPENCAGE_KEY, timeout=10)
    return geolocator.geocode(place)

lat = lon = None
if location:
    loc = geocode_location(location)
    if loc:
        lat, lon = loc.latitude, loc.longitude
    else:
        st.error("No se pudo encontrar la ciudad.")

# Mostrar el mapa si hay coordenadas
if lat and lon:
    modo_poligono = st.toggle("Modo polígono", value=st.session_state.get("modo_poligono", False))
    st.session_state.modo_poligono = modo_poligono

    if not st.session_state.modo_poligono:
        st.subheader("Haz clic en el mapa para seleccionar coordenadas precisas")
        m = folium.Map(location=[lat, lon], zoom_start=13)
        folium.LatLngPopup().add_to(m)
        
        # Mostrar el mapa en la app
        map_data = st_folium(m, height=500, width=700)

        # Extraer lat/lon del clic
        if map_data.get("last_clicked"):
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            st.success(f"Coordenadas seleccionadas: {clicked_lat:.5f}, {clicked_lon:.5f}")
            parametros['ll'] = f"{str(clicked_lat)},{str(clicked_lon)}"
            del(parametros['near'])  # Eliminar 'near' si se usa 'll'
            radio = st.number_input("Radio en metros a considerar (máxmio 100.000m):", min_value=1, max_value=100000, value=22000)
            parametros['radius'] = radio
    
    
    if st.session_state.modo_poligono:
        st.subheader("Dibuja el polígono en el mapa")
        m = folium.Map(location=[lat, lon], zoom_start=13)
        # Agregar plugin para dibujar polígonos
        from folium.plugins import Draw
        draw = Draw(
                draw_options={
                    "polyline": False,
                    "rectangle": False,
                    "circle": False,
                    "marker": False,
                    "circlemarker": False,
                    "polygon": {"shapeOptions": {"color": "#6bc2e5"}}
                },
                edit_options={"edit": True}
            )
        draw.add_to(m)
        
        # Mostrar mapa con Streamlit-folium
        map_data = st_folium(m, height=500, width=700)

        # Extraer coordenadas del polígono dibujado (si hay)
        if map_data and "all_drawings" in map_data:
            drawings = map_data["all_drawings"]
            if drawings:
                polygon_coords = None
                for feature in drawings:
                    if feature["geometry"]["type"] == "Polygon":
                        raw_coords = feature["geometry"]["coordinates"][0]  # Primer anillo del polígono
                        polygon_coords = [(lat, lon) for lon, lat in raw_coords]  # Convertir a (lat, lon)
                        st.session_state['polygon_coords'] = polygon_coords
                        break
                
                if polygon_coords:
                    # Calcular centroide y radio
                    poly_shapely = Polygon([(lat, lon) for lat, lon in polygon_coords])
                    centroide = poly_shapely.centroid
                    centro_lat, centro_lon = centroide.x, centroide.y

                    # Calcular radio como la distancia máxima del centroide a un vértice
                    radio = max(
                        geodesic((centro_lat, centro_lon), (lat, lon)).meters
                        for lat, lon in polygon_coords
                    )
                    parametros['radius'] = int(radio)
                    parametros['ll'] = f"{str(centro_lat)},{str(centro_lon)}"
                    del(parametros['near'])

def get_poi_photo(fsq_id):
    url = f"https://api.foursquare.com/v3/places/{fsq_id}/photos"
    headers = {
        "Authorization": API_KEY,
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        photos = response.json()
        if photos:
            # Tomamos la primera foto disponible
            p = photos[0]
            return f"{p['prefix']}original{p['suffix']}"
    return None  # Si no hay fotos o error

def get_poi_tips(fsq_id):
    url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
    headers = {
        "Authorization": API_KEY,
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        tips = response.json()
        if tips:
            return tips
    return None  # Si no hay fotos o error

def generar_malla_centros(centro, radio_m, step_m):
    lat, lon = centro
    radios_grados = step_m / 111320  # ~ metros por grado

    n_puntos = int(radio_m // step_m)
    centros = []
    for dx in range(-n_puntos, n_puntos + 1):
        for dy in range(-n_puntos, n_puntos + 1):
            new_lat = lat + dy * radios_grados
            new_lon = lon + dx * (radios_grados / np.cos(np.radians(lat)))
            if geodistance((lat, lon), (new_lat, new_lon)).meters <= radio_m:
                centros.append((new_lat, new_lon))
    return centros

def buscar_pois_malla(api_key, centro, radio_m, step_m=250, query=None):
    headers = {"Authorization": api_key}
    centros = generar_malla_centros(centro, radio_m, step_m)
    
    all_pois = {}
    for lat, lon in centros:
        if len(all_pois) >= limit:
            break  # Ya tenemos suficientes POIs, salir del bucle

        params = {
            "ll": f"{lat},{lon}",
            "radius": step_m,
            "limit": 50,
        }
        if query:
            params["query"] = query
        response = requests.get("https://api.foursquare.com/v3/places/search", headers=headers, params=params)
        if response.status_code == 200:
            for poi in response.json().get("results", []):
                all_pois[poi["fsq_id"]] = poi
                if len(all_pois) >= limit:
                    break  # Salir del for interno si alcanzamos el límite

    return list(all_pois.values())[:limit]

# Radio de la Tierra en kilometros
R_EARTH_KM = 6371.0
# Radio de la Tierra en metros
R = 6371000 

def flatten_poi(poi):
        """Aplanar un POI para exportación conservando la mayoría de datos."""
        main_geo = poi.get("geocodes", {}).get("main", {})
        lat = main_geo.get("latitude")
        lon = main_geo.get("longitude")

        return {
            "fsq_id": poi.get("fsq_id"),
            "name": poi.get("name"),
            "categories": json.dumps(poi.get("categories", []), ensure_ascii=False),
            "closed_bucket": poi.get("closed_bucket"),
            "distance": poi.get("distance"),
            "latitude": lat,
            "longitude": lon,
            "link": poi.get("link"),
            "timezone": poi.get("timezone"),
            "chains": json.dumps(poi.get("chains", []), ensure_ascii=False),
            "location": json.dumps(poi.get("location", {}), ensure_ascii=False),
            "related_places": json.dumps(poi.get("related_places", {}), ensure_ascii=False),
        }

def _normalize_labels(raw):
    """
    Normaliza fsq_category_labels en una lista de strings.
    Acepta: list, str con ['a','b'], str sin comillas "[A > B]", None.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None]
    if isinstance(raw, str):
        s = raw.strip()
        # intento parsear literal si tiene comillas
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        # si empieza y termina con [] quitarlas y separar por comas (fallback)
        if s.startswith('[') and s.endswith(']'):
            inner = s[1:-1].strip()
            if inner == "":
                return []
            # si hay comas, dividir, si no, tomar entero
            if ',' in inner:
                parts = [p.strip().strip('\'"') for p in inner.split(',') if p.strip()]
                return parts
            else:
                return [inner]
        # caso general: devolver el string tal cual
        return [s]
    # fallback para otros tipos
    return [str(raw)]

def _haversine_mask_partition(part, lat0, lon0, radius_km):
    """
    Devuelve una Series booleana indicando si fila está dentro del radio.
    Se asume part es un pandas.DataFrame (map_partitions).
    """
    mask = pd.Series(False, index=part.index)
    if ("latitude" not in part.columns) or ("longitude" not in part.columns):
        return mask

    lat = pd.to_numeric(part["latitude"], errors="coerce")
    lon = pd.to_numeric(part["longitude"], errors="coerce")
    valid = lat.notnull() & lon.notnull()
    if not valid.any():
        return mask

    lat_vals = np.radians(lat[valid].values.astype(float))
    lon_vals = np.radians(lon[valid].values.astype(float))
    lat0r = math.radians(lat0)
    lon0r = math.radians(lon0)

    dlat = lat_vals - lat0r
    dlon = lon_vals - lon0r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(lat_vals) * np.sin(dlon / 2) ** 2
    dist = 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))

    mask.loc[valid.index] = dist <= radius_km
    return mask

def _query_mask_partition(part, tokens, search_cols):
    """
    Construye una máscara booleana por partición buscando tokens en search_cols.
    tokens: lista de strings (ya en minúsculas).
    """
    mask = pd.Series(False, index=part.index)
    if part.shape[0] == 0:
        return mask

    for col in search_cols:
        if col not in part.columns:
            continue
        col_ser = part[col].fillna("")

        # caso especial: categories que pueden ser listas o strings con []
        if col == "fsq_category_labels":
            # aplica normalize_labels por elemento (vectorizado con apply)
            def cat_match(x):
                labs = _normalize_labels(x)
                if not labs:
                    return False
                for lab in labs:
                    lab_l = str(lab).lower()
                    for t in tokens:
                        # match por substring o match con la parte más específica después de ">"
                        if t in lab_l:
                            return True
                        if ">" in lab_l:
                            most_specific = lab_l.split(">")[-1].strip()
                            if t == most_specific:
                                return True
                return False

            mask = mask | col_ser.apply(cat_match)
        else:
            # otros campos: transformamos a string y buscamos tokens (regex OR)
            ser_str = col_ser.astype(str).str.lower()
            # escape tokens para regex, y buscar cualquiera
            pattern = "|".join(re.escape(t) for t in tokens)
            # na=False para evitar nulos
            mask = mask | ser_str.str.contains(pattern, na=False)

    return mask

def filtrar_pois(df, location=None, query=None, search_cols=None):
    """
    Filtra un Dask DataFrame de POIs.

    - location: (lat_center, lon_center, radius_km). Si None, no filtra por ubicación.
    - query: string con tokens separados por coma (ej: "cafe, restaurant, +34")
    - search_cols: lista de columnas en las que buscar query. Si None, usamos columnas por defecto.

    Retorna:
      - Dask DataFrame (lazy) si limit is None
      - pandas.DataFrame si limit es int (ya computado)
    """
    # columnas por defecto para búsqueda libre
    if search_cols is None:
        search_cols = [
            "fsq_category_labels", "name", "address", "locality", "region",
            "postcode", "tel", "website", "email"
        ]

    filtrado = df

    # 1) Filtrado por bounding box aproximado (rápido)
    if location:
        lat0, lon0, radius_km = location
        # aproximación para delta lat/lon
        delta_lat = radius_km / 110.574
        # evitar cos(90°) cuando lat0 ~ +/-90
        cos_lat = math.cos(math.radians(lat0))
        cos_lat = max(cos_lat, 1e-6)
        delta_lon = radius_km / (111.320 * cos_lat)

        lat_min, lat_max = lat0 - delta_lat, lat0 + delta_lat
        lon_min, lon_max = lon0 - delta_lon, lon0 + delta_lon

        filtrado = filtrado[
            (filtrado["latitude"] >= lat_min) &
            (filtrado["latitude"] <= lat_max) &
            (filtrado["longitude"] >= lon_min) &
            (filtrado["longitude"] <= lon_max)
        ]

        # 1b) refinar por haversine (vectorizado por partición)
        # usamos map_partitions para aplicar la máscara por partición de forma eficiente
        def _keep_in_circle(part):
            m = _haversine_mask_partition(part, lat0, lon0, radius_km)
            return part[m]
        filtrado = filtrado.map_partitions(_keep_in_circle, meta=filtrado._meta)

    # 2) Filtrado por query (tokens)
    if query:
        tokens = [t.strip().lower() for t in query.split(",") if t.strip()]
        if tokens:
            def _filter_part_by_query(part):
                m = _query_mask_partition(part, tokens, search_cols)
                return part[m]
            filtrado = filtrado.map_partitions(_filter_part_by_query, meta=filtrado._meta)

    return filtrado.compute()

def filtrar_pois_fast(dataset_path, location=None, query=None, search_cols=None):
    if search_cols is None:
        search_cols = [
            "fsq_category_labels", "name", "address", "locality", "region",
            "postcode", "tel", "website", "email"
        ]

    filters = []

    # Bounding box → pushdown en lectura
    if location:
        lat0, lon0, radius_km = location
        delta_lat = radius_km / 110.574
        cos_lat = max(math.cos(math.radians(lat0)), 1e-6)
        delta_lon = radius_km / (111.320 * cos_lat)

        lat_min, lat_max = lat0 - delta_lat, lat0 + delta_lat
        lon_min, lon_max = lon0 - delta_lon, lon0 + delta_lon

        filters = [
            ("latitude", ">=", lat_min),
            ("latitude", "<=", lat_max),
            ("longitude", ">=", lon_min),
            ("longitude", "<=", lon_max),
        ]

    df = dd.read_parquet(dataset_path,
                         storage_options={"token": hf_token},
                         filters=filters)

    # Filtrado por query (rápido)
    if query:
        tokens = [t.strip().lower() for t in query.split(",") if t.strip()]
        if tokens:
            pattern = "|".join(tokens)
            mask = False
            for col in search_cols:
                mask = mask | df[col].str.lower().str.contains(pattern, na=False)
            df = df[mask]

    return df  # Dask DataFrame lazy

# Convertimos la cadena en lista de diccionarios
def parse_categories(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except (ValueError, SyntaxError):
        return []

def parse_dict(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except (ValueError, SyntaxError):
        return {}

col_izquierda, col_centrado, col_derecha = st.columns([1, 1, 1])
with col_centrado:
    if location:
        if st.button("📡 Llamar a la API"):
            url = "https://api.foursquare.com/v3/places/search"
            headers = {"accept": "application/json", 
                       "Authorization": API_KEY
                       }
            params = parametros
            center_lat = float(parametros['ll'].split(",")[0])
            center_lon = float(parametros['ll'].split(",")[1])
            radio = parametros['radius']
            radius_km = radio / 1000
            if limit > 50:
                # Usamos la malla de puntos para obtener más de 50 resultados
                data = buscar_pois_malla(API_KEY, centro=(center_lat, center_lon), radio_m=radio, query=query)

            else:
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    data = data.get("results", [])
                else:
                    response_json = response.json()
                    st.error(f"Error en la llamada a la API: {response_json['message']}")
                    st.error(f"{response}")

            # Mostrar número de POIs encontrados
            st.success(f"🔎 Se han encontrado {len(data)} lugares mediante la llamada a la API.")

            campos_finales = ['fsq_id', 
                              'name',
                              'category_id',
                              'category_name',
                              'category_short_name',
                              'category_plural_name',
                              'category_icon_prefix',
                              'category_icon_suffix',
                              'latitude',
                              'longitude',
                              'address',
                              'locality',
                              'postcode',
                              'admin_region',
                              'country',
                              'region',
                              'date_created',
                              'date_refreshed',
                              'date_closed',
                              'tel',
                              'website',
                              'email',
                              'facebook_id',
                              'instagram',
                              'twitter',
                              'closed_bucket'
                              ]
            if data is not None and len(data) > 0:
                # API -> DataFrame
                df_api = pd.DataFrame([flatten_poi(poi) for poi in data])

                # Aplicamos el parse a las columnas
                df_api['categories_parsed'] = df_api['categories'].apply(parse_categories)
                df_api['location_parsed'] = df_api['location'].apply(parse_dict)

                # Obtenemos los campos de los diccionarios de las columnas de categories y location:
                df_api['category_id'] = df_api['categories_parsed'].apply(lambda x: x[0].get('id', np.nan) if isinstance(x, list) and len(x) > 0 else None)
                df_api['category_name'] = df_api['categories_parsed'].apply(lambda x: x[0].get('name', np.nan) if isinstance(x, list) and len(x) > 0 else None)
                df_api['category_short_name'] = df_api['categories_parsed'].apply(lambda x: x[0].get('short_name', np.nan) if isinstance(x, list) and len(x) > 0 else None)
                df_api['category_plural_name'] = df_api['categories_parsed'].apply(lambda x: x[0].get('plural_name', np.nan) if isinstance(x, list) and len(x) > 0 else None)
                df_api['category_icon_prefix'] = df_api['categories_parsed'].apply(lambda x: x[0].get('icon', np.nan).get('prefix', np.nan) if isinstance(x, list) and len(x) > 0 else None)
                df_api['category_icon_suffix'] = df_api['categories_parsed'].apply(lambda x: x[0].get('icon', np.nan).get('suffix', np.nan) if isinstance(x, list) and len(x) > 0 else None)

                df_api['address'] = df_api['location_parsed'].apply(lambda x: x.get('formatted_address', np.nan) if isinstance(x, dict) and len(x) > 0 else None)
                df_api['country'] = df_api['location_parsed'].apply(lambda x: x.get('country', np.nan) if isinstance(x, dict) and len(x) > 0 else None)
                if 'dma' in df_api.loc[0, 'location_parsed']:
                    df_api['admin_region'] = df_api['location_parsed'].apply(lambda x: x.get('dma', np.nan) if isinstance(x, dict) and len(x) > 0 else None)
                if 'admin_region' in df_api.loc[0, 'location_parsed']:
                    df_api['admin_region'] = df_api['location_parsed'].apply(lambda x: x.get('admin_region', np.nan) if isinstance(x, dict) and len(x) > 0 else None)
                df_api['locality'] = df_api['location_parsed'].apply(lambda x: x.get('locality', np.nan) if isinstance(x, dict) and len(x) > 0 else None)
                df_api['postcode'] = df_api['location_parsed'].apply(lambda x: x.get('postcode', np.nan) if isinstance(x, dict) and len(x) > 0 else None)
                df_api['region'] = df_api['location_parsed'].apply(lambda x: x.get('region', np.nan) if isinstance(x, dict) and len(x) > 0 else None)

                # Creamos las columnas que le faltan al dataset de la API para unificarlo más tarde con el de HugginGFace
                df_api['date_created'] = None
                df_api['date_refreshed'] = None
                df_api['date_closed'] = None
                df_api['tel'] = None
                df_api['website'] = None
                df_api['email'] = None
                df_api['facebook_id'] = None
                df_api['instagram'] = None
                df_api['twitter'] = None
                
                df_api = df_api[campos_finales]
            
            else:
                df_api = pd.DataFrame(columns=campos_finales)
            # Convertimos a JSON
            data = json.loads(df_api.to_json(orient="records", force_ascii=False))
            st.session_state['data'] = data  # Guarda datos crudos

            # Si mediante la llamada a la API se han encontrado menos POIs de los solicitados 
            # verificamos si en el dataset de Foursquare hay más
            if len(data) < limit or not data:
                len_df = st.session_state.get('len_HF_dataset', None)
                st.write(f"Vamos a intentar completar los POIs obtenidos llamando a la API con los {len_df} POIs del dataset de Foursquare")
                # Filstrar los POIs del dataset
                dataset_path = "hf://datasets/foursquare/fsq-os-places/release/dt=2025-08-07/places/parquet/*.parquet"

                df_dataset = filtrar_pois_fast(dataset_path, location=(center_lat, center_lon, radius_km), query=query)

                st.success(f"Se han encontrado {len(df_dataset)} POIs adicionales en el dataset de Foursquare.")

                # Traemos solo 'limit' filas desde dask (rápido)
                df_dataset_pd = df_dataset.head(limit)

                # Renombramos columnas para unificar
                df_dataset_pd = df_dataset_pd.rename(columns={'fsq_place_id': 'fsq_id'})

                # Nos quedamos con el primer id de las categorias de los pois
                df_dataset_pd["category_id"] = df_dataset_pd["fsq_category_ids"].apply(lambda x: x[0] if x is not None and len(x) > 0 else None)

                # Creamos la columa categoria a partir del id
                df_dataset_pd = df_dataset_pd.merge(
                    categories_fsq[['category_id', 'category_name']],
                    on='category_id',
                    how='left'
                )

                # Obtenemos los otros campos de categoría a partir de los disponibles en el dataset de la api
                df_dataset_pd['category_id'] = df_dataset_pd['category_id'].astype(str)
                df_api['category_id'] = df_api['category_id'].astype(str)
                df_dataset_pd = df_dataset_pd.merge(
                    df_api[['category_id', 'category_short_name', 'category_plural_name', 'category_icon_suffix', 'category_icon_prefix']].drop_duplicates(subset=['category_id']),
                    on='category_id',
                    how='left'
                )

                df_dataset_pd['closed_bucket'] = np.nan
                df_dataset_pd = df_dataset_pd[campos_finales]

                # Concatenamos
                df_combined = pd.concat([df_api, df_dataset_pd], ignore_index=True)

                # Quitamos duplicados por fsq_id
                df_combined = df_combined.drop_duplicates(subset='fsq_id')

                # Nos aseguramos que los POIs finales están en la zona deseada
                st.write(f"Vamos a filtrar los {len(df_combined)} POIs obtenidos para asegurarnos que cumplen las peticiones.")
                # Convertir a radianes los puntos de la circunferencia
                polygon_coords = st.session_state.get('polygon_coords', None)
                if polygon_coords is not None:
                    polygon = Polygon([(lat, lon) for lat, lon in polygon_coords])
                    # Filtrar POIs dentro del polígono
                    df_combined['inside'] = df_combined.apply(lambda row: polygon.contains(Point(row['latitude'], row['longitude'])), axis=1)
                    df_combined = df_combined[df_combined['inside']].drop(columns=['inside'])

                else:
                    lat1 = np.radians(center_lat)
                    lon1 = np.radians(center_lon)
                    lat2 = np.radians(df_combined['latitude'])
                    lon2 = np.radians(df_combined['longitude'])
                    # Fórmula de Haversine
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                    distance = R * c  # Distancia en metros
                    # Filtrar
                    df_combined = df_combined[distance <= radio]

                categorias =  parametros.get('categories', None)
                if categorias:
                    # Filtramos por categorías si se han indicado
                    categorias_list = [cat.strip().lower() for cat in categorias.split(",") if cat.strip()]
                    df_combined = df_combined[df_combined['category_name'].str.lower().isin(categorias_list)]
                
                # Limitamos filas
                df_combined = df_combined.head(limit)
                st.session_state['dataframe'] = df_combined  # Guardamos el dataframe en sesión

                # Convertimos a JSON
                combined_json = json.loads(df_combined.to_json(orient="records", force_ascii=False))

                st.success(f"Finalmente se obtienen {len(df_combined)} POIs que cumplen las peticiones.")

                # Guardamos en sesión
                st.session_state['data'] = combined_json


# Visualización de los resultados
data = st.session_state.get('data', None)
if data:
    # Mostrar número de POIs encontrados
    st.success(f"🔎 Se han encontrado un total de {len(data)} lugares.")
    # Mostrar los POIs en la interfaz
    locations = []
    limite_fotos = 10
    if limit < limite_fotos:
        limite_fotos = limit
    if len(data) < limite_fotos:
        limite_fotos = len(data)
    st.subheader(f"Mostrando los primeros {limite_fotos} POIs encontrados:")
    for poi in data[:limite_fotos]:
        name = poi.get("name", "Sin nombre")
        fsq_id = poi.get("fsq_id")

        coords = {'latitude': poi.get('latitude', None), 'longitude': poi.get('longitude', None)}
        category_name = poi.get("category_name", "Sin categoría")
        icon_prefix = poi.get("category_icon_prefix", None)
        icon_suffix = poi.get("category_icon_suffix", None)
        if icon_prefix and icon_suffix:
            icon_url = f"{icon_prefix}bg_64{icon_suffix}"  # tamaño bg_64
        else:
            icon_url = None
        
        # Obtener imagen real
        photo_url = get_poi_photo(fsq_id)

        # Mostrar info del POI
        cols = st.columns([1, 9])
        with cols[0]:
            if photo_url:
                st.image(photo_url, width=60)
            elif icon_url:
                st.image(icon_url, width=40)
            else:
                st.image("https://via.placeholder.com/40", width=40)
        with cols[1]:
            st.markdown(f"**{name}**  \n*{category_name}*")
        
        # Guardar info para el mapa
        if coords:
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            if lat and lon:
                locations.append({
                    "name": name,
                    "category": category_name,
                    "lat": lat,
                    "lon": lon,
                    "photo": photo_url
                })
        st.session_state['locations'] = locations  # Guarda locs
        

col_mapa, col_analytics = st.columns(2)

# Botón para mostrar el mapa: solo cambia la flag
locations = st.session_state.get('locations', None)
# Solo mostramos el botón para mostrar mapa si NO está activada la flag show_map
if locations and not st.session_state.get('show_map', False):
    with col_mapa:
        if st.button("🗺️ Ver estos POIs en el mapa"):
            st.session_state['analytics'] = False
            st.session_state['show_map'] = True
# Botón para ocultar mapa
if st.session_state.get('show_map'):
    with col_mapa:
        if st.button("❌ Ocultar mapa"):
            st.session_state['show_map'] = False

# Mostrar el MAPA automáticamente si la flag está activa
if st.session_state.get('show_map'):
    if locations:
        center = [locations[0]["lat"], locations[0]["lon"]]
        m = folium.Map(location=center, zoom_start=14)

        for loc in locations:
            popup_html = f"""
            <b>{loc['name']}</b><br>
            <i>{loc['category']}</i><br>
            <img src="{loc['photo']}" width="100">
            """
            folium.Marker(
                location=[loc["lat"], loc["lon"]],
                popup=folium.Popup(popup_html, max_width=200)
            ).add_to(m)

        st_folium(m, height=500, width=700)
    else:
        st.warning("No hay ubicaciones válidas para mostrar.")

data = st.session_state.get('data', None)


# Botón para la ANALISTICA DE DATOS
if data and not st.session_state['analytics'] and not st.session_state['show_map']:
    with col_analytics:
        if st.button("📊 Ver analítica de los datos obtenidos"):
            st.session_state['show_map'] = False
            st.session_state['analytics'] = True
if st.session_state['analytics']:
    st.subheader("Análisis de los datos obtenidos")
    # Mostrar número total de POIs
    st.write(f"Total de POIs encontrados: {len(data)}")

    # --------------------
    # Extraer categorías
    # --------------------
    category_counts = {}
    rows = []
    for poi in data:
        name = poi.get("name", "Sin nombre")
        fsq_id = poi.get("fsq_id", "")
        lat = poi.get("latitude", None)
        lon = poi.get("longitude", None)
        
        cat_name = poi.get("category_name", "Sin categoría")
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        rows.append({
            "fsq_id": fsq_id,
            "name": name,
            "category": cat_name,
            "latitude": lat,
            "longitude": lon
        })
    
    df = pd.DataFrame(rows)

    # --------------------
    # Gráfico de barras
    # --------------------
    if category_counts:
        st.write("### Categorías más comunes:")
        df_cat = pd.DataFrame(category_counts.items(), columns=["Categoría", "Cantidad"])
        df_cat = df_cat.sort_values(by="Cantidad", ascending=False).head(10)

        fig_bar = px.bar(
            df_cat,
            x="Categoría",
            y="Cantidad",
            title="Top 10 categorías más frecuentes",
            color="Cantidad",
            text="Cantidad"
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar)

    else:
        st.write("No se encontraron categorías.")

    # --------------------
    # Gráfico de dispersión (si hay coordenadas)
    # --------------------
    if not df.dropna(subset=["latitude", "longitude"]).empty:
        st.write("### Mapa de dispersión de POIs:")
        fig_map = px.scatter_mapbox(
            df.dropna(subset=["latitude", "longitude"]),
            lat="latitude",
            lon="longitude",
            hover_name="name",
            hover_data=["category"],
            color="category",
            zoom=12,
            height=500
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map)
    else:
        st.write("No hay coordenadas geográficas disponibles para mostrar el mapa.")


# Permitir descarga
if data:
    st.subheader("📥 Descargar datos")

    flattened = [flatten_poi(poi) for poi in data]
    df_full = st.session_state.get('dataframe', None)

    # Crear geometría para GeoDataFrame
    df_full["geometry"] = df_full.apply(lambda row: Point(row["longitude"], row["latitude"]) 
                                        if pd.notnull(row["latitude"]) and pd.notnull(row["longitude"]) 
                                        else None, axis=1)
    gdf = gpd.GeoDataFrame(df_full, geometry="geometry", crs="EPSG:4326")

    col1, col2, col3, col4 = st.columns(4)

    # JSON crudo (ya lo tienes)
    json_data = json.dumps(data, indent=4, ensure_ascii=False)
    with col1:
        st.download_button("📄 Descargar como JSON", 
                           data=json_data, 
                           file_name="foursquare_data.json", 
                           mime="application/json")

    # CSV
    csv_io = io.StringIO()
    data_df = st.session_state.get('dataframe', None)
    if data_df is not None:
        data_df.to_csv(csv_io, index=False)
    else:
        df_full.drop(columns=["geometry"]).to_csv(csv_io, index=False)
    with col2:
        st.download_button("📑 Descargar como CSV", 
                           data=csv_io.getvalue(), 
                           file_name="foursquare_data.csv", 
                           mime="text/csv")

    # GeoJSON
    with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tmpfile:
        geojson_path = tmpfile.name  # guardamos solo el nombre, lo cerramos al salir del bloque
    # Ahora escribimos con geopandas
    gdf.to_file(geojson_path, driver="GeoJSON", encoding="utf-8")
    # Leemos el contenido como bytes
    with open(geojson_path, "rb") as f:
        geojson_bytes = f.read()
    # Eliminamos el archivo temporal (opcional)
    os.remove(geojson_path)
    with col3:
        st.download_button("🌍 Descargar como GeoJSON", 
                           data=geojson_bytes, 
                           file_name="foursquare_data.geojson", 
                           mime="application/geo+json")

    # Shapefile
    with tempfile.TemporaryDirectory() as tmpdir:
        shapefile_path = os.path.join(tmpdir, "pois.shp")
        gdf.to_file(shapefile_path, driver="ESRI Shapefile", encoding="utf-8")

        zip_path = os.path.join(tmpdir, "pois_shapefile.zip")
        # with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zipf:
            for filename in os.listdir(tmpdir):
                if filename.startswith("pois"):
                    # zipf.write(os.path.join(tmpdir, filename), arcname=filename)
                    file_path = os.path.join(tmpdir, filename)
                    # Crear información de ZIP forzando ZIP64
                    info = zipfile.ZipInfo(filename)
                    info.file_size = os.path.getsize(file_path)
                    
                    with open(file_path, "rb") as f:
                        zipf.writestr(info, f.read())
        with open(zip_path, "rb") as f:
            zip_bytes = f.read()
        with col4:
            st.download_button(
                "🗂️ Descargar como Shapefile (ZIP)",
                data=zip_bytes,
                file_name="foursquare_data_shapefile.zip",
                mime="application/zip")







