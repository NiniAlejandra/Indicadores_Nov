# ESIP - Dashboard de Indicadores  
### *Resumen Cualitativo y Visualización Interactiva*  
**Período:** Enero → Septiembre 2025  
**Última actualización:** 30 de octubre de 2025  
**Herramienta:** Streamlit + Plotly + Pandas  

---

## OBJETIVO
Proporcionar una **visión cualitativa inmediata** del desempeño de los indicadores clave de ESIP mediante un **semáforo de colores** y **gráficos interactivos**.

---

## SEMÁFORO CUALITATIVO

| Color | Valoración | Significado |
|-------|------------|-----------|
| <span style="color:green">**VERDE**</span> | **Alto** | Excelente / Óptimo |
| <span style="color:orange">**NARANJA**</span> | **Medio** | Aceptable / En mejora |
| <span style="color:red">**ROJO**</span> | **Bajo** | Crítico / Requiere acción |

---

## REGLAS DE VALORACIÓN

### 1. **Indicadores en Porcentaje**
| Rango | Valoración | Color |
|------|------------|-------|
| **≥ 90%** | **Alto** | Verde |
| **80% – 89%** | **Medio** | Naranja |
| **< 80%** | **Bajo** | Rojo |

> **Excepción especial**:  
> - **Tasa de Accidentes Laborales**  
> - **Prevalencia de enfermedades**  
> → **0% = Alto (Verde)**  
> → **> 1% = Bajo (Rojo)**

---

### 2. **Indicadores Numéricos** 
Se evalúa la **tendencia mensual** según el **tipo**:

| Tipo | Mejor si... | Cambio mensual |
|------|-------------|----------------|
| **POS** | Sube | +5% → Alto | ±5% → Medio | < -5% → Bajo |
| **NEG** | Baja | -10% → Alto | ±10% → Medio | > +10% → Bajo |
| **NEU** | N/A | Solo se muestra → "N/A" |

> **Flechas de tendencia**:  
> - **up** → Aumentó  
> - **down** → Disminuyó  
> - **right** → Sin cambio (±2%)

---

## ESTRUCTURA DE LA APP
