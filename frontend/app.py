import streamlit as st
import pandas as pd
import requests
import altair as alt
import pydeck as pdk
import time
from geopy.geocoders import Nominatim
from collections import Counter

st.set_page_config(page_title="Perfume Market Dashboard", layout="wide")
st.title("🌸 Perfume Market Dashboard")

# Cargar datos
response = requests.get("http://backend:8000/perfumes")
if response.status_code != 200:
    st.error("Error al cargar los datos.")
    st.stop()

data = response.json()
df = pd.DataFrame(data)

# Limpieza de datos
df['Rating Value'] = df['Rating Value'].str.replace(',', '.', regex=False)
df['rating'] = pd.to_numeric(df['Rating Value'], errors='coerce').round(2)
df['votes'] = df['Rating Count'].astype(int)
df['brand'] = df['Brand'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.strip()
df['brand'] = df['brand'].str.replace(r'\s+', ' ', regex=True).str.title().str.strip()
df['sales'] = (df['rating'] * df['votes'] * 100).astype(int)

# Crear pestañas
tab_datos, tab_graficos, tab_mapa, tab_modelo = st.tabs(["📋 Datos", "📊 Gráficos", "🗺️ Mapa","🤖 Modelo"])

# --- TAB 1: Datos ---
with tab_datos:
    st.subheader("📋 Base de datos")
    st.dataframe(df)

    st.markdown("### 📈 Estadísticas generales")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Perfumes", len(df))
    col2.metric("Average Rating", f"{round(df['rating'].mean(), 2)} / 5")
    col3.metric("Total Estimated Sales", f"{int(df['sales'].sum()):,}")

# --- TAB 2: Gráficos ---
with tab_graficos:
    alt.themes.enable('fivethirtyeight')

    # Acordes más comunes
    st.markdown("### 💐 Acordes más comunes")
    if any(col.startswith('mainaccord') for col in df.columns):
        accord_columns = [col for col in df.columns if col.startswith('mainaccord')]
        df['accords'] = df[accord_columns].apply(lambda row: row.dropna().tolist(), axis=1)
        accord_list = df['accords'].explode().dropna()
        top_accords = pd.DataFrame(Counter(accord for accord_list in df['accords'] if isinstance(accord_list, list) for accord in accord_list).most_common(10), columns=['accord', 'count'])
        chart4 = alt.Chart(top_accords).mark_bar().encode(
            x='count:Q', y=alt.Y('accord:N', sort='-x'), tooltip=['accord', 'count']
        ).properties(height=300)
        st.altair_chart(chart4, use_container_width=True)

    # Distribución de Ratings
    st.markdown("### ⭐ Distribución de Ratings")
    chart2 = alt.Chart(df).mark_bar().encode(
        x=alt.X("rating:Q", bin=True), y="count():Q", tooltip=["count()"]
    ).properties(height=300)
    st.altair_chart(chart2, use_container_width=True)

    

    # Ventas por Marca
    st.markdown("### 🏷️ Ventas por Marca")
    sales_by_brand = df.groupby("brand")["sales"].sum().reset_index().sort_values(by="sales", ascending=False).head(10)
    chart1 = alt.Chart(sales_by_brand).mark_bar().encode(
        x='sales:Q', y=alt.Y('brand:N', sort='-x'), color='sales:Q', tooltip=['brand', 'sales']
    ).properties(height=400)
    st.altair_chart(chart1, use_container_width=True)

    

# --- TAB 3: Mapa ---
with tab_mapa:
    st.markdown("### 🌍 Mapa de Perfumes por País")
    country_count = df['Country'].dropna().value_counts().reset_index()
    country_count.columns = ['Country', 'count']

    geolocator = Nominatim(user_agent="perfume_dashboard")
    coords_cache = {}

    def get_coordinates(country):
        if country in coords_cache:
            return coords_cache[country]
        try:
            location = geolocator.geocode(country, timeout=10)
            if location:
                coords = [location.latitude, location.longitude]
                coords_cache[country] = coords
                time.sleep(1)
                return coords
        except:
            return None

    country_count[['lat', 'lon']] = country_count['Country'].apply(get_coordinates).apply(
        lambda x: pd.Series(x) if x else pd.Series([None, None])
    )
    country_count.dropna(subset=['lat', 'lon'], inplace=True)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=country_count,  # Usa datos serializables
        get_position='[lon, lat]',
        get_radius='count*100',  # Usa la columna calculada
        get_fill_color='[180, 0, 200, 140]',
        pickable=True
    )

    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.2, pitch=30)

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "{Country}: {count} perfumes"}
    ))
# ---------- TAB 4: RECOMENDADOR ----------
import joblib
import numpy as np

with tab_modelo:
    st.markdown("## 🤖 Recomendador de Rating de Perfume")

    model = joblib.load("best_rf_perfume_fast.pkl")
    model_columns = joblib.load("model_columns_perfume.pkl")

    def build_input_row_dict(country, gender, tops, middles, bases, accords, model_columns):
        d = {col: 0 for col in model_columns}
        if f"Country_{country}" in d: d[f"Country_{country}"] = 1
        if f"Gender_{gender}" in d: d[f"Gender_{gender}"] = 1
        for note in tops: d[f"Note_Top_{note.strip().lower()}"] = 1
        for note in middles: d[f"Note_Middle_{note.strip().lower()}"] = 1
        for note in bases: d[f"Note_Base_{note.strip().lower()}"] = 1
        for acc in accords: d[f"Accord_{acc.strip().lower()}"] = 1
        return d

    def recommend_one(base_dict, current_rating, all_opts, items, prefix, model, model_columns):
        rows, labels = [], []
        for opt in all_opts:
            if opt not in items:
                d = base_dict.copy()
                if items:
                    d.pop(f"{prefix}_{items[-1].strip().lower()}", None)
                d[f"{prefix}_{opt}"] = 1
                rows.append(d); labels.append(opt)
        if not rows: return None, current_rating
        X_try = pd.DataFrame(rows).reindex(columns=model_columns, fill_value=0)
        preds = model.predict(X_try)
        i = np.argmax(preds)
        return labels[i], preds[i]

    # ———————————————
    # Limpiar y extraer etiquetas únicas para multiselect
    def get_unique_note_values(series):
        return sorted(set(
            note.strip().lower()
            for entry in series.dropna()
            for note in str(entry).split(',')
            if note.strip()
        ))

    with st.form("perfume_form"):
        # Opciones limpias
        country_options = sorted(df['Country'].dropna().unique()) if 'Country' in df.columns else []
        gender_options = sorted(df['Gender'].dropna().unique()) if 'Gender' in df.columns else ["Male", "Female", "Unisex"]

        top_note_options = get_unique_note_values(df['Top']) if 'Top' in df.columns else []
        middle_note_options = get_unique_note_values(df['Middle']) if 'Middle' in df.columns else []
        base_note_options = get_unique_note_values(df['Base']) if 'Base' in df.columns else []

        # Acordes desde columnas mainaccord
        if any(col.startswith('mainaccord') for col in df.columns):
            accord_cols = [col for col in df.columns if col.startswith('mainaccord')]
            accord_series = df[accord_cols].astype(str).apply(lambda x: ','.join(x), axis=1)
            accord_options = get_unique_note_values(accord_series)
        else:
            accord_options = []

        # Campos del formulario
        country = st.selectbox("🌍 País de lanzamiento", country_options)
        gender = st.selectbox("🧍 Género", gender_options)
        tops = st.multiselect("Notas Top:", top_note_options, default=top_note_options[:2])
        middles = st.multiselect("Notas Middle:", middle_note_options, default=middle_note_options[:2])
        bases = st.multiselect("Notas Base:", base_note_options, default=base_note_options[:2])
        accords = st.multiselect("Acordes Principales:", accord_options, default=accord_options[:3])

        # Botón para enviar
        submitted = st.form_submit_button("📈 Calcular rating y recomendaciones")

        if submitted:
            base_dict = build_input_row_dict(country, gender, tops, middles, bases, accords, model_columns)
            X_base = pd.DataFrame([base_dict]).reindex(columns=model_columns, fill_value=0)
            current_rating = model.predict(X_base)[0]
            st.metric("⭐ Rating estimado", f"{current_rating:.2f}")

            # Recomendaciones
            all_top = [c.replace("Note_Top_", "") for c in model_columns if c.startswith("Note_Top_")]
            all_mid = [c.replace("Note_Middle_", "") for c in model_columns if c.startswith("Note_Middle_")]
            all_base = [c.replace("Note_Base_", "") for c in model_columns if c.startswith("Note_Base_")]
            all_acc = [c.replace("Accord_", "") for c in model_columns if c.startswith("Accord_")]

            st.subheader("🔧 Cambios sugeridos")
            for name, items, all_opts, pref in [
                ("Top Note", tops, all_top, "Note_Top"),
                ("Middle Note", middles, all_mid, "Note_Middle"),
                ("Base Note", bases, all_base, "Note_Base"),
                ("Accord", accords, all_acc, "Accord")
            ]:
                change, pred = recommend_one(base_dict, current_rating, all_opts, items, pref, model, model_columns)
                if change:
                    st.write(f"- Sustituir un {name} por *{change}* → rating estimado: {pred:.2f}")
                else:
                    st.write(f"- No se encontró mejora cambiando un {name.lower()}.")
