from flask import Flask, render_template, request
from dash import Dash, html, dcc, callback, Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import datetime as dt

# Inicializar Flask
app = Flask(__name__)

# Hablar con la API de yfinance
def obtener_datos_tesla(ticker, first_date, last_date):
    empresa = yf.Ticker(ticker)
    empresa_nombre = empresa.info['longName']
    df = empresa.history(
        start=first_date,
        end=last_date,
        interval="1d"
    )

    # Resetear el índice para tener la fecha como columna
    df = df.reset_index()

    # Crear la columna Adj Close (en este caso será igual a Close ya que yfinance ya ajusta los precios)
    df['Adj Close'] = df['Close']
    # Calcular los campos adicionales que necesitas
    df['daily_change_%'] = df['Close'].pct_change() * 100
    df['volatility'] = df['High'] - df['Low']
    df['daily_avg'] = (df['High'] + df['Low']) / 2
    df['distance'] = df['Close'] - df['Open']
    df['result'] = df['distance'].apply(lambda x: 'up' if x > 0 else 'down')
    
    # Calcular volume_relative (comparado con la media móvil de 20 días)
    df['volume_relative'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    return df, empresa_nombre


#VALIDACIONES y CALCULOS

def validar_columnas(df):
    """Validar que existan las columnas requeridas"""
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'distance', 'result', 'volatility', 'daily_avg', 'daily_change_%', 'volume_relative']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"La columna {column} no existe en el CSV. "
                f"Columnas disponibles: {df.columns.tolist()}"
            )
    return required_columns

def calcular_metricas_volatilidad(df):
    """Calcular métricas de volatilidad"""
    # Crear una copia del DataFrame para no modificar el original
    df_vol = df.copy()
    
    # Calcular retornos diarios
    df_vol['returns'] = df_vol['Close'].pct_change()
    
    # Volatilidad histórica (desviación estándar de retornos)
    volatilidad_historica = df_vol['returns'].std() * (252 ** 0.5)  # Anualizada
    
    # True Range y Average True Range (ATR)
    df_vol['TR'] = pd.DataFrame({
        'HL': df_vol['High'] - df_vol['Low'],
        'HC': abs(df_vol['High'] - df_vol['Close'].shift(1)),
        'LC': abs(df_vol['Low'] - df_vol['Close'].shift(1))
    }).max(axis=1)
    atr = df_vol['TR'].rolling(window=14).mean().iloc[-1]
    
    # Volatilidad de Parkinson (basada en High-Low)
    df_vol['HL_volatility'] = np.log(df_vol['High'] / df_vol['Low'])
    parkinson_volatility = (1 / (4 * np.log(2))) * df_vol['HL_volatility'].rolling(window=30).std() * np.sqrt(252)
    
    # Calcular volatilidad móvil de 20 días para el gráfico
    df_vol['volatility_20'] = df_vol['returns'].rolling(window=20).std() * np.sqrt(252) * 100
    
    return {
        'df_vol': df_vol,  # DataFrame con todos los cálculos
        'volatilidad_historica': round(volatilidad_historica * 100, 2),
        'atr': round(atr, 2),
        'parkinson_volatility': round(parkinson_volatility.iloc[-1] * 100, 2)
    }

#FUNCIONES
def crear_tabla(df, required_columns):
    """Crear tabla con Plotly"""
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=required_columns,
            line_color='darkslategray',
            fill_color='#0369a1',
            align='left'
        ),
        cells=dict(
            values=[df[col] for col in required_columns],
            line_color='darkslategray',
            fill_color='rgba(0,0,0,0)',
            align='left'
        ))
    ])
    
    fig.update_layout(
        width=750, 
        height=300,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def crear_grafica_lineas(df, width, height, x, y):
    """Crear gráfica de líneas"""
    fig_line = px.line(df, x=x, y=y, title='Gráfica de Líneas: Fecha vs Apertura')
    
    # Cambiar color de la línea
    fig_line.update_traces(line_color='#0369a1')  # Color azul cielo
    
    fig_line.update_layout(
        width=width, 
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=35, r=35, t=35, b=35),
    )
    return fig_line

def crear_grafica_barras(df, x, y, width, height):
    """Crear gráfica de barras"""
    fig_bar = px.bar(df, x=x, y=y, title='Gráfica de Barras: Fecha vs Apertura')
    
    # Cambiar color de las barras
    fig_bar.update_traces(marker_color='#0369a1')  # Color azul cielo
    
    fig_bar.update_layout(
        width=width, 
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=35, r=35, t=35, b=35),
    )
    return fig_bar

def crear_grafica_resultados(result_count):
    """Crear gráfica de barras para mostrar conteo de ups y downs"""
    df_results = pd.DataFrame({
        'Resultado': ['Subidas', 'Bajadas'],
        'Cantidad': [result_count['up'], result_count['down']]
    })
    
    # Crear gráfica con colores personalizados
    fig = px.pie(df_results, 
                 values='Cantidad',
                 names='Resultado',
                 title='Distribución de Subidas vs Bajadas',
                 color='Resultado',
                 color_discrete_map={
                     'Subidas': '#0284c7',  # Verde
                     'Bajadas': '#0369a1'   # Rojo
                 })
    
    fig.update_layout(
        width=720,
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False
    )
    
    return fig

def crear_grafica_pastel(df, width, height):
    """Crear gráfica de pastel que muestra la distribución de volumen por mes"""
    # Convertir la columna Date a datetime si no lo está ya
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Agregar datos por mes y calcular el volumen total
    df_monthly = df.groupby(df['Date'].dt.strftime('%B'))['Volume'].sum().reset_index()
    
    # Crear gráfica de pastel
    fig_pastel = px.pie(
        df_monthly, 
        values='Volume', 
        names='Date',
        title='Distribución de Volumen por Mes',
        color_discrete_sequence=['#0284c7', '#0369a1', '#075985', '#0c4a6e']  # Diferentes tonos de azul
    )
    fig_pastel.update_layout(
        width=width,
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True
    )
    return fig_pastel

def crear_grafica_velas(df, width, height):
    """Crear gráfica de velas (candlestick)"""
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='#0284c7',  # Color para velas alcistas
        decreasing_line_color='#0369a1'   # Color para velas bajistas
    )])
    
    fig.update_layout(
        title='Gráfico de Velas - TESLA',
        width=width,
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=35, r=35, t=35, b=35),
        yaxis_title='Precio',
        xaxis_title='Fecha'
    )
    
    return fig

def crear_grafica_volatilidad(df, width, height):
    """Crear gráfica de volatilidad"""
    fig = go.Figure()
    
    # Añadir línea de volatilidad
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['volatility_20'],
        name='Volatilidad 20 días',
        line=dict(color='#0369a1')
    ))
    
    fig.update_layout(
        title='Volatilidad Histórica (20 días)',
        width=width,
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=35, r=35, t=35, b=35),
        yaxis_title='Volatilidad (%)',
        xaxis_title='Fecha'
    )
    
    return fig


#RUTAS
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
            primera_fecha = request.form.get('primera_fecha', '2024-01-01')
            segunda_fecha = request.form.get('segunda_fecha', '2024-12-31')
            ticker = request.form.get('ticker', 'TSLA')
    try:
        primera_fecha = pd.to_datetime(primera_fecha).strftime('%Y-%m-%d')
        segunda_fecha = pd.to_datetime(segunda_fecha).strftime('%Y-%m-%d')


        # Leer el archivo CSV y convertir fechas correctamente
        df, nombre_empresa = obtener_datos_tesla(ticker, primera_fecha, segunda_fecha)

        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')  # Cambiado el formato
        
        # Obtener fechas mínima y máxima del dataset
        primera_fecha = df['Date'].min().strftime('%Y-%m-%d')
        segunda_fecha = df['Date'].max().strftime('%Y-%m-%d')
        
        # Obtener fechas del formulario si es POST
            
        # Filtrar el DataFrame por fechas
        mask = (df['Date'] >= primera_fecha) & (df['Date'] <= segunda_fecha)
        df_filtrado = df.loc[mask].copy()
        

        # Calculos Extra
        fecha_actual = dt.datetime.now().strftime('%Y-%m-%d')
        max_open = round(df_filtrado['Open'].max(), 2)
        min_open = round(df_filtrado['Open'].min(), 2)
        max_close = round(df_filtrado['Close'].max(), 2)
        min_close = round(df_filtrado['Close'].min(), 2)
        
        # Contar subidas y bajadas
        result_count = df_filtrado['result'].value_counts()

        

        metricas_vol = calcular_metricas_volatilidad(df_filtrado)
        required_columns = validar_columnas(df_filtrado)
        
        # Crear gráficas con df_filtrado
        fig_tabla = crear_tabla(df_filtrado, required_columns)
        fig_line = crear_grafica_lineas(df_filtrado, 750, 300, 'Date', 'Open')
        fig_bar = crear_grafica_barras(df_filtrado, 'Date', 'Open', 720, 300)
        fig_pastel = crear_grafica_pastel(df_filtrado, 600, 300)
        fig_volatilidad = crear_grafica_volatilidad(metricas_vol['df_vol'], 1000, 300)
        fig_velas = crear_grafica_velas(df_filtrado, 1000, 300)
        fig_resultados = crear_grafica_resultados(result_count)
        
        
        
        # Convertir a HTML
        grafica_tabla_plotly = fig_tabla.to_html(full_html=False)
        grafica_velas_plotly = fig_velas.to_html(full_html=False)
        grafica_volatilidad_plotly = fig_volatilidad.to_html(full_html=False)
        grafica_line_plotly = fig_line.to_html(full_html=False)
        grafica_bar_plotly = fig_bar.to_html(full_html=False)
        grafica_pastel_plotly = fig_pastel.to_html(full_html=False)
        grafica_resultados_plotly = fig_resultados.to_html(full_html=False)
        

        
        context = {
            'estado': 'success',
            'mensaje': 'Archivo CSV leído correctamente',
            'grafica_velas_plotly': grafica_velas_plotly,
            'grafica_volatilidad_plotly': grafica_volatilidad_plotly,
            'volatilidad_historica': metricas_vol['volatilidad_historica'],
            'atr': metricas_vol['atr'],
            'parkinson_volatility': metricas_vol['parkinson_volatility'],
            'fecha_actual': fecha_actual,
            'max_open': max_open,
            'min_open': min_open,
            'max_close': max_close,
            'min_close': min_close,
            'grafica_lineas_plotly': grafica_line_plotly,
            'grafica_barras_plotly': grafica_bar_plotly,
            'grafica_pastel_plotly': grafica_pastel_plotly,
            'tabla_plotly': grafica_tabla_plotly,
            'primera_fecha': primera_fecha,
            'segunda_fecha': segunda_fecha,
            'nombre_empresa': nombre_empresa,
            'ticker': ticker,
            'grafica_resultados_plotly': grafica_resultados_plotly
        }
        
        return render_template('index.html', **context)
        
    except Exception as e:
        print(f"Error detallado: {str(e)}")
        return render_template('index.html', estado='error', mensaje=str(e))

if __name__ == '__main__':
    app.run(debug=True)
