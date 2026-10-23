
import pandas as pd

from src.data.market_data import fetch_market_data


def main():
    # Prueba 1: descargar dos activos
    prices = fetch_market_data(
        tickers=["SPY", "MSFT"],
        start_date="2023-01-01",
        end_date="2025-01-01",
    )

    print("\n--- PRUEBA 1: DESCARGA ---")
    print(prices.head())
    print(prices.tail())

    # Prueba 2: verificar estructura
    print("\n--- PRUEBA 2: ESTRUCTURA ---")
    print("Tipo:", type(prices))
    print("Dimensiones:", prices.shape)
    print("Columnas:", prices.columns.tolist())
    print("Índice:", type(prices.index))

    assert isinstance(prices, pd.DataFrame)
    assert set(prices.columns) == {"SPY", "MSFT"}
    assert not prices.empty
    assert prices.index.is_monotonic_increasing

    # Prueba 3: verificar precios
    print("\n--- PRUEBA 3: CALIDAD ---")
    print("Valores faltantes:")
    print(prices.isna().sum())

    print("Precios <= 0:", (prices <= 0).sum().sum())

    assert (prices.dropna() > 0).all().all()

    # Prueba 4: descargar un solo ticker
    single = fetch_market_data(
        tickers=["SPY"],
        start_date="2023-01-01",
        end_date="2025-01-01",
    )

    print("\n--- PRUEBA 4: UN SOLO TICKER ---")
    print(single.head())

    assert isinstance(single, pd.DataFrame)
    assert single.columns.tolist() == ["SPY"]

    print("\nTodas las comprobaciones finalizaron.")


if __name__ == "__main__":
    main()
