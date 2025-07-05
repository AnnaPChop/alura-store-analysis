import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def cargar_datos(archivos):
    """Carga y combina datos de múltiples archivos CSV.
    
    Args:
        archivos (list): Lista de rutas a los archivos CSV de las tiendas.
        
    Returns:
        DataFrame: Dataset combinado con datos de todas las tiendas.
    """
    dfs = []
    for i, archivo in enumerate(archivos, 1):
        df = pd.read_csv(archivo)
        df['Tienda'] = f'Tienda {i}'
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def limpiar_datos(df):
    """Limpia y prepara los datos para el análisis.
    
    Args:
        df (DataFrame): Dataset original.
        
    Returns:
        DataFrame: Dataset limpio y preparado.
    """
    # Convertir fecha a datetime
    df['Fecha de Compra'] = pd.to_datetime(df['Fecha de Compra'], format='%d/%m/%Y')
    
    # Calcular total de venta
    df['Total Venta'] = df['Precio'] + df['Costo de envío']
    
    # Calcular porcentaje de costo de envío
    df['Porcentaje_Envio'] = (df['Costo de envío'] / df['Precio']) * 100
    
    return df

def calcular_metricas_tienda(df):
    """Calcula métricas clave por tienda.
    
    Args:
        df (DataFrame): Dataset limpio.
        
    Returns:
        DataFrame: Métricas calculadas por tienda.
    """
    metricas = pd.DataFrame(index=df['Tienda'].unique())
    
    # Métricas financieras
    metricas['Ingresos_Totales'] = df.groupby('Tienda')['Total Venta'].sum()
    metricas['Promedio_Venta'] = df.groupby('Tienda')['Total Venta'].mean()
    
    # Métricas de satisfacción
    metricas['Calificacion_Promedio'] = df.groupby('Tienda')['Calificación'].mean()
    metricas['Porcentaje_5_Estrellas'] = df[df['Calificación'] == 5].groupby('Tienda').size() / df.groupby('Tienda').size() * 100
    
    # Métricas operacionales
    metricas['Eficiencia_Envio'] = 100 - df.groupby('Tienda')['Porcentaje_Envio'].mean()
    
    return metricas

def calcular_crecimiento(df):
    """Calcula tasa de crecimiento por tienda.
    
    Args:
        df (DataFrame): Dataset limpio.
        
    Returns:
        Series: Tasa de crecimiento por tienda.
    """
    df['Año_Mes'] = df['Fecha de Compra'].dt.to_period('M')
    ventas_mensuales = df.groupby(['Tienda', 'Año_Mes'])['Total Venta'].sum().reset_index()
    
    def calcular_tasa(grupo):
        return ((grupo['Total Venta'].iloc[-1] - grupo['Total Venta'].iloc[0]) / grupo['Total Venta'].iloc[0]) * 100
    
    return ventas_mensuales.groupby('Tienda').apply(calcular_tasa)

def generar_ranking(df):
    """Genera ranking final de tiendas basado en múltiples métricas.
    
    Args:
        df (DataFrame): Dataset limpio.
        
    Returns:
        DataFrame: Ranking final de tiendas con puntajes normalizados.
    """
    # Obtener métricas base
    ranking = calcular_metricas_tienda(df)
    
    # Agregar tasa de crecimiento
    ranking['Tasa_Crecimiento'] = calcular_crecimiento(df)
    
    # Normalizar métricas
    for columna in ranking.columns:
        ranking[columna] = (ranking[columna] - ranking[columna].min()) / (ranking[columna].max() - ranking[columna].min())
    
    # Calcular puntaje final con pesos
    ranking['Puntaje_Final'] = (
        0.30 * (ranking['Ingresos_Totales'] + ranking['Promedio_Venta'])/2 +
        0.25 * (ranking['Calificacion_Promedio'] + ranking['Porcentaje_5_Estrellas'])/2 +
        0.25 * ranking['Eficiencia_Envio'] +
        0.20 * ranking['Tasa_Crecimiento']
    )
    
    return ranking.sort_values('Puntaje_Final', ascending=False)

def configurar_visualizacion():
    """Configura el estilo de las visualizaciones."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette('husl')
    plt.rcParams['figure.figsize'] = (12, 8)

def graficar_ingresos_tienda(df):
    """Genera gráfico de barras de ingresos por tienda.
    
    Args:
        df (DataFrame): Dataset limpio.
    """
    ingresos_tienda = df.groupby('Tienda')['Total Venta'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=ingresos_tienda.index, y=ingresos_tienda.values)
    plt.title('Ingresos Totales por Tienda')
    plt.xlabel('Tienda')
    plt.ylabel('Ingresos Totales')
    plt.savefig('../visualizations/ingresos_por_tienda.png')
    plt.close()

def graficar_distribucion_ventas(df):
    """Genera gráfico circular de distribución de ventas por categoría.
    
    Args:
        df (DataFrame): Dataset limpio.
    """
    ventas_categoria = df.groupby('Categoría del Producto')['Total Venta'].sum()
    
    plt.figure(figsize=(12, 8))
    plt.pie(ventas_categoria.values, labels=ventas_categoria.index, autopct='%1.1f%%')
    plt.title('Distribución de Ventas por Categoría')
    plt.savefig('../visualizations/distribucion_ventas_categoria.png')
    plt.close()

def graficar_calificaciones_ingresos(df):
    """Genera gráfico de dispersión entre calificaciones e ingresos.
    
    Args:
        df (DataFrame): Dataset limpio.
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='Calificación', y='Total Venta', hue='Tienda')
    plt.title('Relación entre Calificaciones e Ingresos')
    plt.xlabel('Calificación')
    plt.ylabel('Total Venta')
    plt.savefig('../visualizations/relacion_calificaciones_ingresos.png')
    plt.close()

def graficar_tendencia_ventas(df):
    """Genera gráfico de líneas de tendencia de ventas en el tiempo.
    
    Args:
        df (DataFrame): Dataset limpio.
    """
    ventas_tiempo = df.groupby(['Fecha de Compra', 'Tienda'])['Total Venta'].sum().reset_index()
    
    plt.figure(figsize=(12, 8))
    for tienda in df['Tienda'].unique():
        data = ventas_tiempo[ventas_tiempo['Tienda'] == tienda]
        plt.plot(data['Fecha de Compra'], data['Total Venta'], label=tienda)
    plt.title('Tendencia de Ventas en el Tiempo')
    plt.xlabel('Fecha')
    plt.ylabel('Total Ventas')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig('../visualizations/tendencia_ventas_tiempo.png')
    plt.close()

def graficar_distribucion_calificaciones(df):
    """Genera boxplot de distribución de calificaciones por tienda.
    
    Args:
        df (DataFrame): Dataset limpio.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='Tienda', y='Calificación')
    plt.title('Distribución de Calificaciones por Tienda')
    plt.xlabel('Tienda')
    plt.ylabel('Calificación')
    plt.savefig('../visualizations/distribucion_calificaciones.png')
    plt.close()

def graficar_correlacion_metricas(df):
    """Genera heatmap de correlación entre métricas clave.
    
    Args:
        df (DataFrame): Dataset limpio.
    """
    metricas_correlacion = df[['Total Venta', 'Precio', 'Costo de envío', 'Calificación', 'Cantidad de cuotas']].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(metricas_correlacion, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlación entre Métricas Clave')
    plt.savefig('../visualizations/correlacion_metricas.png')
    plt.close()