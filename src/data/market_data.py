
import yfinance as yf
import pandas as pd


def fetch_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Descarga precios históricos ajustados para los activos solicitados.

    Returns:
        DataFrame con fechas en el índice y tickers en las columnas.

    Raises:
        ValueError: si los parámetros son inválidos o faltan activos.
        RuntimeError: si la descarga falla o no devuelve datos.
    """

    # 1. Validar los parámetros
    tickers = [ticker.strip().upper() for ticker in tickers]

    if not tickers:
        raise ValueError("Debe proporcionar al menos un ticker.")

    if len(tickers) != len(set(tickers)):
        raise ValueError("La lista contiene tickers duplicados.")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start >= end:
        raise ValueError("start_date debe ser anterior a end_date.")

    # 2. Descargar los precios ajustados
    print(f"Descargando {tickers} desde {start_date} hasta {end_date}...")

    try:
        raw = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
    except Exception as exc:
        raise RuntimeError(f"Error al descargar datos: {exc}") from exc

    if raw.empty:
        raise RuntimeError("La descarga no devolvió datos.")

    # 3. Extraer Close y normalizar el formato
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]

        elif "Close" in raw.columns.get_level_values(1):
            prices = raw.xs("Close", axis=1, level=1)

        else:
            raise RuntimeError("No se encontró la columna Close.")

    else:
        if "Close" not in raw.columns:
            raise RuntimeError("No se encontró la columna Close.")

        # Descarga de un solo ticker
        prices = raw[["Close"]].rename(
            columns={"Close": tickers[0]}
        )

    # 4. Asegurar un formato uniforme
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices.columns = prices.columns.astype(str).str.upper()
    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")

    # 5. Verificar que estén todos los activos solicitados
    missing_tickers = set(tickers) - set(prices.columns)

    if missing_tickers:
        raise ValueError(
            f"No se obtuvieron datos para: {sorted(missing_tickers)}"
        )

    prices = prices[tickers]

    # 6. Eliminar fechas en las que ningún activo tiene precio
    prices = prices.dropna(how="all")

    if prices.empty:
        raise RuntimeError("No hay precios válidos.")

    # 7. Validar precios disponibles
    invalid_prices = prices.notna() & (prices <= 0)

    if invalid_prices.any().any():
        raise ValueError("Se encontraron precios iguales o inferiores a cero.")

    # Importante: no rellenamos los NaN automáticamente.
    return prices
