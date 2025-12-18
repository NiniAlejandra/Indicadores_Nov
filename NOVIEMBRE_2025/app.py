import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================
st.set_page_config(layout="wide", page_title="Indicadores ESIP 2025")

# Logo en la parte superior derecha
col_logo, col_title = st.columns([1, 10])
with col_logo:
    try:
        st.image('logo_esip_clear.png', width=100)
    except:
        pass

with col_title:
    st.markdown("<h1 style='text-align: center;'>Indicadores ESIP SAS ESP - ENE - NOV 2025</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Reporte de gestión enero a noviembre 2025</p>", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE LIMPIEZA Y PREPARACIÓN DE DATOS
# ============================================================================

def limpiar_valor_numerico(valor):
    """Limpia valores numéricos eliminando comas y convirtiendo a float"""
    if pd.isna(valor) or valor == '' or str(valor).strip() == '':
        return np.nan
    if isinstance(valor, str):
        valor_str = str(valor).strip()
        if valor_str in ['#DIV/0!', '#ERROR!', '-', 'N/A', 'N/A']:
            return np.nan
        # Eliminar comas, espacios y símbolos de moneda
        valor_limpio = valor_str.replace(',', '').replace('$', '').replace(' ', '').replace('COP', '')
        # Manejar valores negativos con guión al inicio
        es_negativo = valor_limpio.startswith('-')
        valor_limpio = valor_limpio.replace('-', '')
        try:
            resultado = float(valor_limpio)
            return -resultado if es_negativo else resultado
        except:
            return np.nan
    try:
        return float(valor)
    except:
        return np.nan

def limpiar_valor_porcentual(valor):
    """Limpia valores porcentuales eliminando % y convirtiendo a float"""
    if pd.isna(valor) or valor == '' or str(valor).strip() == '':
        return np.nan
    if isinstance(valor, str):
        valor_str = str(valor).strip()
        if valor_str in ['#DIV/0!', '#ERROR!', '-', 'N/A']:
            return np.nan
        # Eliminar %, comas y espacios
        valor_limpio = valor_str.replace('%', '').replace(',', '').replace(' ', '')
        try:
            return float(valor_limpio)
        except:
            return np.nan
    try:
        return float(valor)
    except:
        return np.nan

def cargar_y_limpiar_datos():
    """Carga y limpia los datos de ambos CSV"""
    try:
        # Cargar archivos CSV
        df_num = pd.read_csv('Ind_num.csv', sep=',', encoding='utf-8')
        df_porc = pd.read_csv('Ind_porc.csv', sep=',', encoding='utf-8')
        
        # Agregar columna de tipo
        df_num['Tipo'] = 'Numéricos'
        df_porc['Tipo'] = 'Porcentuales'
        
        # Lista de meses
        meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre']
        
        # Limpiar columnas numéricas en df_num
        for mes in meses:
            if mes in df_num.columns:
                df_num[mes] = df_num[mes].apply(limpiar_valor_numerico)
        
        if 'Total' in df_num.columns:
            df_num['Total'] = df_num['Total'].apply(limpiar_valor_numerico)
        
        # Limpiar columnas porcentuales en df_porc
        for mes in meses:
            if mes in df_porc.columns:
                df_porc[mes] = df_porc[mes].apply(limpiar_valor_porcentual)
        
        if 'Total' in df_porc.columns:
            df_porc['Total'] = df_porc['Total'].apply(limpiar_valor_porcentual)
        
        # Eliminar columna Valoración si existe
        if 'Valoración' in df_num.columns:
            df_num = df_num.drop(columns=['Valoración'])
        if 'Valoración' in df_porc.columns:
            df_porc = df_porc.drop(columns=['Valoración'])
        
        # Concatenar ambos DataFrames
        df = pd.concat([df_num, df_porc], ignore_index=True)
        
        # Si hay filas duplicadas (mismo ID, Área, Indicador), consolidar
        # Agrupar por ID, Área, Indicador y tomar el primer valor no nulo de cada columna
        columnas_agrupar = ['ID', 'Área', 'Indicador']
        
        # Función para consolidar valores (tomar el primero no nulo)
        def consolidar_serie(serie):
            valores_no_nulos = serie.dropna()
            if len(valores_no_nulos) > 0:
                return valores_no_nulos.iloc[0]
            return np.nan
        
        # Agrupar y consolidar todas las columnas excepto las de agrupación
        columnas_para_consolidar = [col for col in df.columns if col not in columnas_agrupar]
        
        # Crear diccionario de agregación
        dict_agg = {col: consolidar_serie for col in columnas_para_consolidar}
        
        # Agrupar y consolidar
        df_consolidado = df.groupby(columnas_agrupar, as_index=False).agg(dict_agg)
        
        # Asegurar que Tipo tenga valores
        if 'Tipo' in df_consolidado.columns:
            df_consolidado['Tipo'] = df_consolidado['Tipo'].fillna('Numéricos')
        
        return df_consolidado, meses
    
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None, None

# ============================================================================
# CARGAR DATOS
# ============================================================================
@st.cache_data
def load_data():
    return cargar_y_limpiar_datos()

df, meses = load_data()

if df is None:
    st.stop()

# ============================================================================
# SIDEBAR CON FILTROS
# ============================================================================
st.sidebar.header("Filtros")

# Filtro por rango de meses
mes_inicio, mes_fin = st.sidebar.slider(
    "Rango de Meses",
    min_value=1,
    max_value=11,
    value=(1, 11),
    format="Mes %d"
)

# Convertir números de mes a nombres
meses_seleccionados = meses[mes_inicio-1:mes_fin]

# No aplicar filtros adicionales, usar todos los datos
df_filtrado = df.copy()

# ============================================================================
# COLORES CORPORATIVOS
# ============================================================================
COLOR_PRINCIPAL = '#003087'
COLOR_VERDE = '#00b140'
COLOR_GRIS = '#4a4a4a'
COLORES = [COLOR_PRINCIPAL, COLOR_VERDE, COLOR_GRIS, '#ff6b6b', '#4ecdc4', '#ffe66d']

# ============================================================================
# FUNCIONES AUXILIARES PARA MÉTRICAS
# ============================================================================

def obtener_total_pqrs(df):
    """Obtiene el total de PQRS recibidas"""
    pqrs = df[df['Indicador'].str.contains('Total Pqrs recibidos al mes', case=False, na=False)]
    if not pqrs.empty and 'Total' in pqrs.columns:
        total = pqrs['Total'].sum()
        return int(total) if not pd.isna(total) else 0
    return 0

def obtener_promedio_luminarias_mantenidas(df):
    """Obtiene el promedio mensual de luminarias mantenidas"""
    mantenimiento = df[df['Indicador'].str.contains('luminarias.*MANTENIMIENTO', case=False, na=False, regex=True)]
    if not mantenimiento.empty:
        valores_mensuales = []
        for mes in meses:
            if mes in mantenimiento.columns:
                valores = mantenimiento[mes].dropna()
                if not valores.empty:
                    valores_mensuales.extend(valores.tolist())
        if valores_mensuales:
            return int(np.mean(valores_mensuales))
    return 0

def obtener_total_metros_construidos(df):
    """Obtiene el total de metros lineales construidos"""
    construccion = df[df['Indicador'].str.contains('CONSTRUIDAS', case=False, na=False)]
    if not construccion.empty and 'Total' in construccion.columns:
        total = construccion['Total'].sum()
        return int(total) if not pd.isna(total) else 0
    return 0

# ============================================================================
# ESTRUCTURA PRINCIPAL CON TABS
# ============================================================================

# Crear lista de tabs solo con las áreas
areas_disponibles = sorted(df['Área'].unique().tolist())
tabs = st.tabs(areas_disponibles)

# ============================================================================
# PESTAÑAS POR ÁREA
# ============================================================================
for idx, area in enumerate(areas_disponibles):
    with tabs[idx]:
        st.subheader(f"Área: {area}")
        
        # Filtrar datos del área
        df_area = df_filtrado[df_filtrado['Área'] == area].copy()
        
        if df_area.empty:
            st.info("No hay datos para la selección actual")
        else:
            # Dividir en dos columnas
            col_izq, col_der = st.columns([0.6, 0.4])
            
            with col_izq:
                st.markdown("### Gráficos")
                
                # Preparar datos para gráfico combinado (numéricos y porcentuales)
                datos_grafico_combinado = []
                
                # Agregar indicadores numéricos
                df_area_numericos = df_area[df_area['Tipo'] == 'Numéricos'].copy()
                if not df_area_numericos.empty:
                    for _, row in df_area_numericos.iterrows():
                        for mes in meses_seleccionados:
                            if mes in row.index and not pd.isna(row[mes]):
                                datos_grafico_combinado.append({
                                    'Indicador': row['Indicador'],
                                    'Mes': mes,
                                    'Valor': row[mes],
                                    'Tipo': 'Numérico'
                                })
                
                # Agregar indicadores porcentuales
                df_area_porcentuales = df_area[df_area['Tipo'] == 'Porcentuales'].copy()
                if not df_area_porcentuales.empty:
                    for _, row in df_area_porcentuales.iterrows():
                        for mes in meses_seleccionados:
                            if mes in row.index and not pd.isna(row[mes]):
                                datos_grafico_combinado.append({
                                    'Indicador': row['Indicador'],
                                    'Mes': mes,
                                    'Valor': row[mes],
                                    'Tipo': 'Porcentual'
                                })
                
                # Gráficos separados para numéricos y porcentuales
                if datos_grafico_combinado:
                    df_plot_combinado = pd.DataFrame(datos_grafico_combinado)
                    
                    # Gráfico para indicadores numéricos
                    df_numericos_plot = df_plot_combinado[df_plot_combinado['Tipo'] == 'Numérico']
                    if not df_numericos_plot.empty:
                        fig_lineas_numericos = px.line(
                            df_numericos_plot,
                            x='Mes',
                            y='Valor',
                            color='Indicador',
                            title=f'Evolución Mensual - Indicadores Numéricos ({area})',
                            markers=True
                        )
                        fig_lineas_numericos.update_layout(
                            xaxis_title="Mes",
                            yaxis_title="Valor",
                            height=400,
                            colorway=COLORES
                        )
                        st.plotly_chart(fig_lineas_numericos, use_container_width=True)
                    
                    # Gráfico para indicadores porcentuales
                    df_porcentuales_plot = df_plot_combinado[df_plot_combinado['Tipo'] == 'Porcentual']
                    if not df_porcentuales_plot.empty:
                        fig_lineas_porcentuales = px.line(
                            df_porcentuales_plot,
                            x='Mes',
                            y='Valor',
                            color='Indicador',
                            title=f'Evolución Mensual - Indicadores Porcentuales ({area})',
                            markers=True
                        )
                        fig_lineas_porcentuales.update_layout(
                            xaxis_title="Mes",
                            yaxis_title="Porcentaje (%)",
                            height=400,
                            colorway=COLORES
                        )
                        st.plotly_chart(fig_lineas_porcentuales, use_container_width=True)
                
                # Gráfico geográfico si hay indicadores por comuna/zona
                indicadores_geograficos = df_area[df_area['Indicador'].str.contains('comuna|zona rural', case=False, na=False, regex=True)]
                if not indicadores_geograficos.empty:
                    datos_geo = []
                    for _, row in indicadores_geograficos.iterrows():
                        if 'Total' in row.index and not pd.isna(row['Total']):
                            datos_geo.append({
                                'Ubicación': row['Indicador'],
                                'Total': row['Total']
                            })
                    
                    if datos_geo:
                        df_geo = pd.DataFrame(datos_geo)
                        # Limpiar nombres para el gráfico
                        df_geo['Ubicacion_limpia'] = df_geo['Ubicación'].str.extract(r'(comuna \d+|zona rural -?\s*\w+)', expand=False)
                        df_geo['Ubicacion_limpia'] = df_geo['Ubicacion_limpia'].fillna(df_geo['Ubicación'])
                        
                        fig_geo = px.bar(
                            df_geo,
                            x='Total',
                            y='Ubicacion_limpia',
                            orientation='h',
                            title=f'Distribución Geográfica - {area}',
                            color='Total',
                            color_continuous_scale=['#003087', '#00b140']
                        )
                        fig_geo.update_layout(
                            xaxis_title="Total",
                            yaxis_title="Ubicación",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig_geo, use_container_width=True)
            
            with col_der:
                st.markdown("### Promedio")
                
                # Calcular promedios de los indicadores del área
                # Calcular promedio por indicador usando los meses seleccionados
                promedios = []
                for _, row in df_area.iterrows():
                    valores_mensuales = []
                    for mes in meses_seleccionados:
                        if mes in row.index and not pd.isna(row[mes]):
                            valores_mensuales.append(row[mes])
                    
                    if valores_mensuales:
                        promedio = np.mean(valores_mensuales)
                        nombre_indicador = row['Indicador'][:50] + '...' if len(row['Indicador']) > 50 else row['Indicador']
                        tipo_indicador = row.get('Tipo', 'Numéricos')
                        
                        # Mostrar métricas de promedio
                        if tipo_indicador == 'Porcentuales' or area in ['SIG', 'Comunicaciones y Marketing', 'Coordinación Administrativa', 
                                                                        'Investigación y Desarrollo Social', 'Jurídica', 'Control de Recursos']:
                            st.metric(nombre_indicador, f"{promedio:.1f}%")
                        else:
                            st.metric(nombre_indicador, f"{promedio:,.0f}")
                
                st.markdown("---")
                
                # Tabla de indicadores del área
                st.markdown("### Indicadores del Área")
                columnas_tabla_area = ['Indicador', 'Formula']
                if 'Total' in df_area.columns:
                    columnas_tabla_area.append('Total')
                columnas_tabla_area.extend(meses_seleccionados)
                
                columnas_disponibles_area = [col for col in columnas_tabla_area if col in df_area.columns]
                st.dataframe(
                    df_area[columnas_disponibles_area],
                    use_container_width=True,
                    hide_index=True
                )
            
            # Expander con datos crudos del área
            with st.expander("Ver datos crudos del área"):
                st.dataframe(df_area, use_container_width=True)

# ============================================================================
# PIE DE PÁGINA
# ============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 12px;'>Dashboard desarrollado para ESIP SAS ESP | Datos: Enero - Noviembre 2025</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: #999; font-size: 10px;'>Desarrollado por Alejandra Valderrama para ESIP SAS ESP</p>",
    unsafe_allow_html=True
)

