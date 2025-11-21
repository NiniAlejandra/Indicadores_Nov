"""
Dashboard Indicadores Internos ESIP S.A.S. E.S.P.
Versión: Octubre 2025
Herramienta: Streamlit + Plotly + Pandas

Para actualizar mensualmente:
1. Reemplazar base_num.csv y base_porc.csv con los nuevos datos
2. Cambiar el título en la línea correspondiente (línea ~50)
3. Ejecutar → deploy en Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Dashboard ESIP - Octubre 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #001F3F;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0074D9;
    }
    .valoracion-alto {
        color: #28a745;
        font-weight: bold;
    }
    .valoracion-medio {
        color: #ff9800;
        font-weight: bold;
    }
    .valoracion-bajo {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Título y logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Intentar cargar el logo, si no existe continuar sin él
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, "logo_esip_clear.png")
    if os.path.exists(logo_path):
        try:
            st.image(logo_path, width=300)
        except Exception as e:
            st.markdown("### ESIP S.A.S. E.S.P.")
    else:
        st.markdown("### ESIP S.A.S. E.S.P.")
    st.markdown('<h1 class="main-header">Dashboard Indicadores Internos ESIP - Octubre 2025</h1>', unsafe_allow_html=True)

# Carga de datos
@st.cache_data
def load_data():
    """Carga y procesa los datos de los archivos CSV"""
    try:
        # Obtener el directorio donde está el script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Cargar archivos CSV usando rutas absolutas basadas en la ubicación del script
        df_num = pd.read_csv(os.path.join(script_dir, "base_num.csv"))
        df_porc = pd.read_csv(os.path.join(script_dir, "base_porc.csv"))
        df_tipo = pd.read_csv(os.path.join(script_dir, "tipo_indicadores.csv"))
        
        # Agregar tipo de indicador
        df_num['Tipo_Indicador'] = 'Numérico'
        df_porc['Tipo_Indicador'] = 'Porcentual'
        
        # Unir ambos dataframes
        df = pd.concat([df_num, df_porc], ignore_index=True)
        
        # Hacer merge con tipo (POS/NEG/NEU)
        # Limpiar espacios en blanco en los nombres de indicadores para mejor matching
        df['Indicador'] = df['Indicador'].str.strip()
        df_tipo['Indicador'] = df_tipo['Indicador'].str.strip()
        df = df.merge(df_tipo[['Indicador', 'Tipo']], on='Indicador', how='left')
        
        # Para indicadores porcentuales sin tipo, asignar tipo basado en el nombre y lógica
        # Los indicadores porcentuales generalmente son POS (mejor si sube) o NEU (neutral)
        # Excepciones: Tasa de Ausentismo, Tasa de Accidentes, Prevalencia de enfermedades son NEG
        mask_sin_tipo = df['Tipo'].isna()
        
        # Indicadores que son NEG (mejor si bajan)
        neg_keywords = ['ausentismo', 'accidentes', 'prevalencia', 'reposición por hurtos']
        df.loc[mask_sin_tipo & df['Indicador'].str.contains('|'.join(neg_keywords), case=False, na=False), 'Tipo'] = 'NEG'
        
        # Indicadores de eficiencia, cumplimiento, eficacia generalmente son POS
        pos_keywords = ['eficiencia', 'eficacia', 'cumplimiento', 'alcance', 'crecimiento', 'avance']
        df.loc[mask_sin_tipo & df['Indicador'].str.contains('|'.join(pos_keywords), case=False, na=False), 'Tipo'] = 'POS'
        
        # El resto sin tipo se marca como NEU
        df.loc[mask_sin_tipo & df['Tipo'].isna(), 'Tipo'] = 'NEU'
        
        # Orden de meses
        month_order = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                      'Julio', 'Agosto', 'Septiembre', 'Octubre']
        
        return df, month_order
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None, None

df, month_order = load_data()

if df is None:
    st.stop()

# Función para limpiar valores numéricos
def clean_numeric(value):
    """Limpia valores numéricos eliminando comas, espacios y símbolos"""
    if pd.isna(value) or value == '' or value == '-' or str(value).strip() == '':
        return np.nan
    if isinstance(value, str):
        # Remover comas, espacios, símbolos de moneda
        value = value.replace(',', '').replace(' ', '').replace('$', '').replace('$', '').strip()
        # Manejar valores negativos con paréntesis o guión
        if value.startswith('(') and value.endswith(')'):
            value = '-' + value[1:-1]
        # Remover % si existe
        is_percentage = '%' in value
        if is_percentage:
            value = value.replace('%', '')
        try:
            result = float(value)
            return result
        except (ValueError, AttributeError):
            return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

# Función para calcular tendencia
def calcular_tendencia(row, month_order):
    """Calcula la tendencia del indicador (↑ ↓ →)"""
    if pd.isna(row.get('Tipo')) or row.get('Tipo') == 'NEU':
        return '→'
    
    # Obtener valores de septiembre y octubre
    sept_val = clean_numeric(row.get('Septiembre', np.nan))
    oct_val = clean_numeric(row.get('Octubre', np.nan))
    
    if pd.isna(sept_val) or pd.isna(oct_val):
        return '→'
    
    if sept_val == 0:
        if oct_val > 0:
            return '↑' if row.get('Tipo') == 'POS' else '↓'
        return '→'
    
    cambio_pct = ((oct_val - sept_val) / abs(sept_val)) * 100
    
    if abs(cambio_pct) < 2:
        return '→'
    elif cambio_pct > 0:
        return '↑'
    else:
        return '↓'

# Aplicar función de tendencia
df['Tendencia'] = df.apply(lambda x: calcular_tendencia(x, month_order), axis=1)

# Sidebar - Filtros
st.sidebar.header("🔍 Filtros")

# Filtro por Área
areas = ['Todas'] + sorted(df['Área'].unique().tolist())
area_seleccionada = st.sidebar.selectbox("Área", areas)

# Filtro por Indicador
if area_seleccionada == 'Todas':
    indicadores_disponibles = sorted(df['Indicador'].unique().tolist())
else:
    indicadores_disponibles = sorted(df[df['Área'] == area_seleccionada]['Indicador'].unique().tolist())

indicadores_seleccionados = st.sidebar.multiselect(
    "Indicador",
    indicadores_disponibles,
    default=indicadores_disponibles  # Mostrar todos los indicadores por defecto
)

# Filtro especial para Atención al Usuario
subcategoria = None
comunas_seleccionadas = None

if area_seleccionada == 'Atención al Usuario':
    subcategorias = ['Todas', 'Comunas', 'Canal de Recepción', 'Subtema', 'Remitido a', 'Otros']
    subcategoria = st.sidebar.selectbox("Subcategoría", subcategorias)
    
    if subcategoria == 'Comunas':
        # Extraer comunas de los indicadores
        comunas = [ind for ind in indicadores_disponibles if 'comuna' in ind.lower()]
        comunas_seleccionadas = st.sidebar.multiselect("Comunas específicas", comunas)

# Aplicar filtros
df_filtrado = df.copy()

if area_seleccionada != 'Todas':
    df_filtrado = df_filtrado[df_filtrado['Área'] == area_seleccionada]

# Si no hay indicadores seleccionados, mostrar todos
if indicadores_seleccionados:
    df_filtrado = df_filtrado[df_filtrado['Indicador'].isin(indicadores_seleccionados)]
# Si la lista está vacía, no filtrar (mostrar todos)

# Filtro adicional por subcategoría
if subcategoria and subcategoria != 'Todas':
    if subcategoria == 'Comunas':
        if comunas_seleccionadas:
            df_filtrado = df_filtrado[df_filtrado['Indicador'].isin(comunas_seleccionadas)]
        else:
            df_filtrado = df_filtrado[df_filtrado['Indicador'].str.contains('comuna', case=False, na=False)]
    elif subcategoria == 'Canal de Recepción':
        df_filtrado = df_filtrado[df_filtrado['Indicador'].str.contains('canal de recepción', case=False, na=False)]
    elif subcategoria == 'Subtema':
        df_filtrado = df_filtrado[df_filtrado['Indicador'].str.contains('subtema', case=False, na=False)]
    elif subcategoria == 'Remitido a':
        df_filtrado = df_filtrado[df_filtrado['Indicador'].str.contains('remitido', case=False, na=False)]
    elif subcategoria == 'Otros':
        # Excluir las otras subcategorías
        mask = (
            ~df_filtrado['Indicador'].str.contains('comuna', case=False, na=False) &
            ~df_filtrado['Indicador'].str.contains('canal de recepción', case=False, na=False) &
            ~df_filtrado['Indicador'].str.contains('subtema', case=False, na=False) &
            ~df_filtrado['Indicador'].str.contains('remitido', case=False, na=False)
        )
        df_filtrado = df_filtrado[mask]

# Resumen Ejecutivo
st.markdown("---")
st.markdown("### 📈 Resumen Ejecutivo")

col1, col2, col3, col4 = st.columns(4)

# Métrica 1: Total Indicadores
total_indicadores = len(df_filtrado)
col1.metric("Total Indicadores", total_indicadores)

# Métrica 2: Promedio cumplimiento porcentuales (solo octubre)
df_porc_oct = df_filtrado[df_filtrado['Tipo_Indicador'] == 'Porcentual'].copy()
if len(df_porc_oct) > 0:
    valores_oct = df_porc_oct['Octubre'].apply(clean_numeric)
    valores_oct = valores_oct[~valores_oct.isna()]
    if len(valores_oct) > 0:
        promedio_porc = valores_oct.mean()
        col2.metric("Promedio Cumplimiento %", f"{promedio_porc:.1f}%")
    else:
        col2.metric("Promedio Cumplimiento %", "N/A")
else:
    col2.metric("Promedio Cumplimiento %", "N/A")

# Métrica 3: % Indicadores en Alto
valoraciones = df_filtrado['Valoración'].value_counts()
alto_count = valoraciones.get('Alto', 0)
porc_alto = (alto_count / total_indicadores * 100) if total_indicadores > 0 else 0
col3.metric("% Indicadores en Alto", f"{porc_alto:.1f}%")

# Métrica 4: Indicadores en estado Bajo
bajo_count = valoraciones.get('Bajo', 0)
col4.metric("Indicadores en Bajo", bajo_count, delta=f"-{bajo_count}" if bajo_count > 0 else None)

st.markdown("---")

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumen Cualitativo", "📈 Tendencias Numéricas", "📉 Indicadores Porcentuales", "📋 Tabla de Datos + README"])

# PESTAÑA 1: Resumen Cualitativo
with tab1:
    st.markdown("### Resumen Cualitativo por Área")
    
    # Agrupar por área
    areas_resumen = df_filtrado.groupby('Área').agg({
        'Indicador': 'count',
        'Valoración': lambda x: {
            'Alto': (x == 'Alto').sum(),
            'Medio': (x == 'Medio').sum(),
            'Bajo': (x == 'Bajo').sum()
        }
    }).reset_index()
    areas_resumen.columns = ['Área', 'Total', 'Valoraciones']
    
    # Mostrar tarjetas por área
    for idx, row in areas_resumen.iterrows():
        area = row['Área']
        total = row['Total']
        val = row['Valoraciones']
        
        st.markdown(f"#### {area}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Indicadores", total)
        col2.metric("✅ Alto", val.get('Alto', 0), delta=None)
        col3.metric("🟡 Medio", val.get('Medio', 0), delta=None)
        col4.metric("❌ Bajo", val.get('Bajo', 0), delta=None)
        st.markdown("---")
    
    # Tabla completa
    st.markdown("### Tabla Completa de Indicadores")
    
    # Preparar datos para la tabla
    df_tabla = df_filtrado.copy()
    
    # Obtener valor de octubre
    df_tabla['Valor Octubre'] = df_tabla['Octubre'].apply(lambda x: str(x) if pd.notna(x) else 'N/A')
    
    # Seleccionar columnas para mostrar
    columnas_tabla = ['Área', 'Indicador', 'Valor Octubre', 'Valoración', 'Tendencia', 'Tipo']
    df_tabla_display = df_tabla[columnas_tabla].copy()
    
    # Ordenar: primero por Área, luego por Valoración (Bajo primero)
    orden_valoracion = {'Bajo': 0, 'Medio': 1, 'Alto': 2, 'N/A': 3}
    df_tabla_display['Orden_Val'] = df_tabla_display['Valoración'].map(orden_valoracion).fillna(3)
    df_tabla_display = df_tabla_display.sort_values(['Área', 'Orden_Val', 'Indicador'])
    df_tabla_display = df_tabla_display.drop('Orden_Val', axis=1)
    
    # Agregar emojis a la columna Valoración antes de aplicar estilos
    df_tabla_display['Valoración_Original'] = df_tabla_display['Valoración'].copy()
    df_tabla_display['Valoración'] = df_tabla_display['Valoración'].apply(
        lambda x: f"✅ {x}" if x == 'Alto' else (f"🟡 {x}" if x == 'Medio' else (f"❌ {x}" if x == 'Bajo' else x))
    )
    
    # Agregar emojis a la tendencia
    df_tabla_display['Tendencia'] = df_tabla_display['Tendencia'].apply(
        lambda x: '↑' if x == '↑' else ('↓' if x == '↓' else '→')
    )
    
    # Eliminar columna auxiliar antes de mostrar
    if 'Valoración_Original' in df_tabla_display.columns:
        df_tabla_display = df_tabla_display.drop('Valoración_Original', axis=1)
    
    st.dataframe(df_tabla_display, use_container_width=True, height=400)

# PESTAÑA 2: Tendencias Numéricas
with tab2:
    st.markdown("### Tendencias de Indicadores Numéricos")
    
    df_num_filtrado = df_filtrado[df_filtrado['Tipo_Indicador'] == 'Numérico'].copy()
    
    if len(df_num_filtrado) == 0:
        st.info("No hay indicadores numéricos para mostrar con los filtros seleccionados.")
    else:
        # Preparar datos para el gráfico
        meses_grafico = month_order
        
        # Crear lista de datos para el gráfico
        fig_data = []
        
        for idx, row in df_num_filtrado.iterrows():
            indicador = row['Indicador']
            # Abreviar nombres largos
            indicador_short = indicador[:40] + "..." if len(indicador) > 40 else indicador
            
            valores = []
            for mes in meses_grafico:
                val = clean_numeric(row.get(mes, np.nan))
                valores.append(val if not pd.isna(val) else None)
            
            fig_data.append({
                'Indicador': indicador,
                'Indicador_Short': indicador_short,
                'Meses': meses_grafico,
                'Valores': valores
            })
        
        # Crear gráfico con Plotly
        fig = go.Figure()
        
        # Colores corporativos
        colores = ['#001F3F', '#0074D9', '#7FDBFF', '#AAAAAA', '#39CCCC', '#3D9970', '#2ECC40', '#FFDC00', '#FF851B', '#FF4136']
        
        for i, data in enumerate(fig_data):
            color = colores[i % len(colores)]
            fig.add_trace(go.Scatter(
                x=data['Meses'],
                y=data['Valores'],
                mode='lines+markers',
                name=data['Indicador_Short'],
                hovertemplate=f"<b>{data['Indicador']}</b><br>" +
                             "Mes: %{x}<br>" +
                             "Valor: %{y:,.0f}<br>" +
                             "<extra></extra>",
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Evolución Mensual de Indicadores Numéricos",
            xaxis_title="Mes",
            yaxis_title="Valor",
            hovermode='closest',
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            xaxis=dict(tickangle=-45),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 3: Indicadores Porcentuales
with tab3:
    st.markdown("### Tendencias de Indicadores Porcentuales")
    
    df_porc_filtrado = df_filtrado[df_filtrado['Tipo_Indicador'] == 'Porcentual'].copy()
    
    if len(df_porc_filtrado) == 0:
        st.info("No hay indicadores porcentuales para mostrar con los filtros seleccionados.")
    else:
        # Preparar datos para el gráfico
        meses_grafico = month_order
        
        # Crear lista de datos para el gráfico
        fig_data = []
        
        for idx, row in df_porc_filtrado.iterrows():
            indicador = row['Indicador']
            # Abreviar nombres largos
            indicador_short = indicador[:40] + "..." if len(indicador) > 40 else indicador
            
            valores = []
            for mes in meses_grafico:
                val = clean_numeric(row.get(mes, np.nan))
                valores.append(val if not pd.isna(val) else None)
            
            fig_data.append({
                'Indicador': indicador,
                'Indicador_Short': indicador_short,
                'Meses': meses_grafico,
                'Valores': valores
            })
        
        # Crear gráfico con Plotly
        fig = go.Figure()
        
        # Colores corporativos
        colores = ['#001F3F', '#0074D9', '#7FDBFF', '#AAAAAA', '#39CCCC', '#3D9970', '#2ECC40', '#FFDC00', '#FF851B', '#FF4136']
        
        for i, data in enumerate(fig_data):
            color = colores[i % len(colores)]
            fig.add_trace(go.Scatter(
                x=data['Meses'],
                y=data['Valores'],
                mode='lines+markers',
                name=data['Indicador_Short'],
                hovertemplate=f"<b>{data['Indicador']}</b><br>" +
                             "Mes: %{x}<br>" +
                             "Valor: %{y:.2f}%<br>" +
                             "<extra></extra>",
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))
        
        # Agregar línea de referencia al 100%
        fig.add_hline(y=100, line_dash="dash", line_color="gray", 
                     annotation_text="100%", annotation_position="right")
        
        fig.update_layout(
            title="Evolución Mensual de Indicadores Porcentuales",
            xaxis_title="Mes",
            yaxis_title="Porcentaje (%)",
            hovermode='closest',
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            xaxis=dict(tickangle=-45),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 4: Tabla de Datos + README
with tab4:
    st.markdown("### Tabla Completa de Datos")
    
    # Preparar tabla con todos los meses
    columnas_tabla_completa = ['ID', 'Área', 'Indicador', 'Tipo_Indicador', 'Tipo', 'Valoración'] + month_order + ['Total']
    columnas_disponibles = [col for col in columnas_tabla_completa if col in df_filtrado.columns]
    df_tabla_completa = df_filtrado[columnas_disponibles].copy()
    
    # Ordenar por Área e Indicador
    df_tabla_completa = df_tabla_completa.sort_values(['Área', 'Indicador'])
    
    st.dataframe(df_tabla_completa, use_container_width=True, height=400)
    
    # Agregar totales al final (suma por mes) - solo para indicadores numéricos
    st.markdown("#### Totales por Mes (Solo Indicadores Numéricos)")
    df_num_totales = df_tabla_completa[df_tabla_completa['Tipo_Indicador'] == 'Numérico'].copy()
    totales_mes = {}
    for mes in month_order:
        if mes in df_num_totales.columns:
            valores_mes = df_num_totales[mes].apply(clean_numeric)
            valores_validos = valores_mes[~valores_mes.isna()]
            if len(valores_validos) > 0:
                totales_mes[mes] = valores_validos.sum()
            else:
                totales_mes[mes] = 0
    
    if totales_mes:
        df_totales = pd.DataFrame([totales_mes])
        st.dataframe(df_totales, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📖 Documentación del Proyecto")
    
    # Leer y mostrar README
    try:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        readme_path = os.path.join(script_dir, "Readme.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
        st.markdown(readme_content, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error al leer el archivo Readme.md: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Dashboard Indicadores Internos ESIP S.A.S. E.S.P. | Versión Octubre 2025</p>
        <p>Desarrollado con Streamlit, Plotly y Pandas</p>
    </div>
    """,
    unsafe_allow_html=True
)

