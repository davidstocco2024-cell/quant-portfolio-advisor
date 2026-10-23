
import pandas as pd

from src.analytics.returns import calculate_returns


# Precios inventados para comprobar la fórmula
prices = pd.DataFrame(
    {
        "SPY": [100, 103, 101, 106],
        "MSFT": [200, 210, 210, 220],
    },
    index=pd.date_range("2025-01-01", periods=4, freq="D"),
)

returns = calculate_returns(prices)

print("Precios:")
print(prices)

print("\nRetornos:")
print(returns)

# SPY: de 100 a 103 = 3%
assert abs(returns["SPY"].iloc[1] - 0.03) < 1e-10

# MSFT: de 200 a 210 = 5%
assert abs(returns["MSFT"].iloc[1] - 0.05) < 1e-10

# El primer retorno no se puede calcular
assert returns.iloc[0].isna().all()

# La cantidad de filas y columnas debe mantenerse
assert returns.shape == prices.shape

print("\nTodas las pruebas finalizaron correctamente.")
