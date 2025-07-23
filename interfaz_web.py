import streamlit as st
import requests
import numpy as np
import json
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon
from geopy.distance import distance as geodistance
from shapely.geometry import box
from geopy.distance import geodesic
import pandas as pd
import plotly.express as px
import io
import geopandas as gpd
import zipfile
from io import BytesIO
import tempfile
import os
import shutil
from dotenv import load_dotenv
load_dotenv()

# Configura tu API Key de Foursquare
API_KEY = os.getenv("FOURSQUARE_API_KEY")

st.title("Descargar Datos de Foursquare")

# Inicializar parametros de sesión
if 'show_map' not in st.session_state:
    st.session_state['show_map'] = False

if 'analytics' not in st.session_state:
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

# Obtener coordenadas de la ciudad
lat = lon = None
if location:
    geolocator = Nominatim(user_agent="foursquare_app")
    location = geolocator.geocode(location)
    if location:
        lat, lon = location.latitude, location.longitude
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
            # st.write(f'drawings: {drawings}')
            if drawings:
                polygon_coords = None
                for feature in drawings:
                    if feature["geometry"]["type"] == "Polygon":
                        raw_coords = feature["geometry"]["coordinates"][0]  # Primer anillo del polígono
                        # st.write(f'raw_coords: {raw_coords}')
                        polygon_coords = [(lat, lon) for lat, lon in raw_coords]  # Convertir a (lat, lon)
                        break
                
                if polygon_coords:
                    # Calcular centroide y radio
                    poly_shapely = Polygon([(lon, lat) for lat, lon in polygon_coords])  # shapely usa (x=lon, y=lat)
                    centroide = poly_shapely.centroid
                    centro_lat, centro_lon = centroide.y, centroide.x

                    # Calcular radio como la distancia máxima del centroide a un vértice
                    radio = max(
                        geodesic((centro_lat, centro_lon), (lat, lon)).meters
                        for lat, lon in polygon_coords
                    )
                    # st.write("Polígono seleccionado:")
                    # st.write(polygon_coords)
                    
                    # Convertir a lat,lng (Foursquare lo quiere así)
                    polygon_param = "~".join([f"{pt[1]},{pt[0]}" for pt in polygon_coords])

                    # Asegurarse de que el polígono esté cerrado (primer punto = último)
                    if polygon_coords[0] != polygon_coords[-1]:
                        polygon_param += f"~{polygon_coords[0][1]},{polygon_coords[0][0]}"
                    # st.write("Parámetro polygon para API:")
                    # st.code(polygon_param)
                    parametros['polygon'] = polygon_param
                    # st.write("Polígono modificado:")
                    # st.write(polygon_param)
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

def buscar_pois_malla(api_key, centro, radio_m, step_m=250, query=None, polygon_coords=None):
    headers = {"Authorization": api_key}
    centros = generar_malla_centros(centro, radio_m, step_m)
    # st.write(centros)

    # Si se proporcionó un polígono, usar shapely para filtrar los centros
    if polygon_coords:
        poligono = Polygon(polygon_coords)
        centros = [p for p in centros if poligono.contains(Point(p[0], p[1]))]
        st.write(centros)
    
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


col_caca1, col_centrado, col_caca_3 = st.columns([1, 1, 1])
with col_centrado:
    if location:
        if st.button("📡 Llamar a la API"):
            url = "https://api.foursquare.com/v3/places/search"
            headers = {"accept": "application/json", "Authorization": API_KEY}
            params = parametros

            if limit > 50:
                # Usar malla con o sin polígono
                if st.session_state.modo_poligono and polygon_coords:
                    data = buscar_pois_malla(API_KEY, centro=(centro_lat, centro_lon), radio_m=radio, query=query, polygon_coords=polygon_coords)
                else:
                    data = buscar_pois_malla(API_KEY, centro=(lat, lon), radio_m=radio, query=query)

            else:
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    # context = data.get("context", {})
                    data = data.get("results", [])
                else:
                    response_json = response.json()
                    st.error(f"Error en la llamada a la API: {response_json['message']}")
            
            st.session_state['data'] = data  # Guarda datos crudos

data = st.session_state.get('data', None)
if data:
        # Mostrar número de POIs encontrados
        st.success(f"🔎 Se han encontrado {len(data)} lugares.")

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
            categories = poi.get("categories", [])
            coords = poi.get("geocodes", {}).get("main", {})
            
            if categories:
                category_name = categories[0].get("name", "Sin categoría")
                icon_prefix = categories[0]["icon"].get("prefix", "")
                icon_suffix = categories[0]["icon"].get("suffix", "")
                icon_url = f"{icon_prefix}bg_64{icon_suffix}"  # tamaño bg_64
            else:
                category_name = "Sin categoría"
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

# Mostrar el mapa automáticamente si la flag está activa
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

# Botón para la analítica de datos
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
        lat = poi.get("geocodes", {}).get("main", {}).get("latitude", None)
        lon = poi.get("geocodes", {}).get("main", {}).get("longitude", None)

        for cat in poi.get("categories", []):
            cat_name = cat.get("name", "Sin categoría")
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

    st.subheader("📥 Descargar datos")

    flattened = [flatten_poi(poi) for poi in data]
    df_full = pd.DataFrame(flattened)

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
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for filename in os.listdir(tmpdir):
                if filename.startswith("pois"):
                    zipf.write(os.path.join(tmpdir, filename), arcname=filename)
        with open(zip_path, "rb") as f:
            zip_bytes = f.read()
        with col4:
            st.download_button(
                "🗂️ Descargar como Shapefile (ZIP)",
                data=zip_bytes,
                file_name="foursquare_data_shapefile.zip",
                mime="application/zip")