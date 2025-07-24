import pandas as pd
import numpy as np

def process_price_data(csv_path: str):
    """
    Reads price data from CSV and calculates for each ticker:
    - 20-day and 60-day moving averages of close price
    - 20-day and 60-day volatility (rolling std dev of daily returns)
    
    Returns a dictionary of DataFrames keyed by ticker.
    """
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values(['ticker', 'date'])
    
    result = {}
    
    for ticker, group in df.groupby('ticker'):
        group = group.copy()
        group['ma_20'] = group['close'].rolling(window=20).mean()
        group['ma_60'] = group['close'].rolling(window=60).mean()
        
        # Calculate daily returns
        group['returns'] = group['close'].pct_change()
        
        # Calculate rolling volatility (std dev of returns)
        group['volatility_20'] = group['returns'].rolling(window=20).std()
        group['volatility_60'] = group['returns'].rolling(window=60).std()
        
        result[ticker] = group.reset_index(drop=True)
    
    return result


def process_price_data_from_json(json_obj):
    """
    Recibe un diccionario de la forma:
    {
      "success": true,
      "data": [ ... ]   # lista de dicts, cada uno una fila de datos
    }
    Calcula para cada ticker:
    - Medias móviles de 20 y 60 días sobre el precio de cierre
    - Volatilidad (std dev rolling) de 20 y 60 días sobre retornos diarios

    Devuelve un dict de DataFrames indexados por ticker.
    """
    # Extraer la lista de datos
    data = [dict(i) for i in json_obj.prices]
    df = pd.DataFrame(data)
    print(df)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    
    result = {}
    
    for ticker, group in df.groupby('ticker'):
        group = group.copy()
        group['ma_20'] = group['close'].rolling(window=20).mean()
        group['ma_60'] = group['close'].rolling(window=60).mean()
        
        # Calcular retornos diarios
        group['returns'] = group['close'].pct_change()
        
        # Calcular volatilidad rolling (std dev de retornos)
        group['volatility_20'] = group['returns'].rolling(window=20).std()
        group['volatility_60'] = group['returns'].rolling(window=60).std()
        
        result[ticker] = group.reset_index(drop=True).dropna()
    
    return result


if __name__ == "__main__":
    df = process_price_data(r"C:\Users\aog13\OneDrive\Escritorio\proyectos\trading_platform_uv_industria\data\prices.csv")