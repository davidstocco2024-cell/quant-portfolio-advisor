
import pandas as pd


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula retornos simples diarios a partir de precios históricos.

    Args:
        prices: DataFrame de precios.
                Cada columna representa un activo.
                El índice contiene las fechas.

    Returns:
        DataFrame de retornos simples.
        El primer retorno de cada serie será NaN,
        porque no existe un precio anterior para calcularlo.
    """

    # 1. Validar los datos de entrada
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices debe ser un DataFrame de pandas.")

    if prices.empty:
        raise ValueError("El DataFrame de precios está vacío.")

    if not prices.index.is_unique:
        raise ValueError("El índice contiene fechas duplicadas.")

    # 2. Ordenar las observaciones cronológicamente
    prices = prices.sort_index()

    # 3. Comprobar que los precios sean numéricos
    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in prices.dtypes
    ):
        raise TypeError("Todas las columnas deben contener valores numéricos.")

    # 4. Comprobar que los precios disponibles sean positivos
    invalid_prices = prices.notna() & (prices <= 0)

    if invalid_prices.any().any():
        raise ValueError("Los precios deben ser positivos.")

    # 5. Calcular los retornos simples
    returns = prices.pct_change(fill_method=None)

    return returns
