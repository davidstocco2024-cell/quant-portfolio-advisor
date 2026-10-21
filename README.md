# 📈 Quant Portfolio Advisor & Risk Analytics

Un sistema cuantitativo de análisis financiero, gestión de riesgos y optimización de portafolios desarrollado en Python. Este proyecto está diseñado para asistir a asesores de inversión y analistas en la toma de decisiones financieras basadas en datos, combinando la Teoría Moderna de Portafolios (MPT) con métricas avanzadas de riesgo.

---

## 🎯 Objetivo del Proyecto

Proporcionar un motor analítico integral que permita:
- Evaluar el rendimiento histórico y la volatilidad ajustada por riesgo de activos financieros (Acciones, CEDEARs, Criptoactivos e Índices).
- Identificar y medir riesgos de cola (*tail risk*) mediante métricas cuantitativas avanzadas.
- Optimizar la asignación de capital (*Asset Allocation*) para construir portafolios eficientes basados en perfiles de riesgo del inversor.

---

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.10+
- **Análisis de Datos:** `pandas`, `numpy`
- **Extracción de Mercado:** `yfinance`
- **Optimización & Riesgo:** `scipy`, `PyPortfolioOpt`
- **Visualización & Dashboard:** `streamlit`, `plotly` *(en desarrollo)*

---

## 🧠 Métricas y Modelos Financieros

El motor analiza y calcula automáticamente:
- **Rendimientos:** Retornos diarios simples y logarítmicos, rendimiento acumulado.
- **Riesgo & Volatilidad:** Volatilidad anualizada, *Maximum Drawdown (MDD)*.
- **Ratios de Performance:** *Sharpe Ratio*, *Sortino Ratio* (penalización exclusiva de volatilidad negativa).
- **Estadística de Distribución:** Sesgo (*Skewness*) y Curtosis (*Kurtosis*) para detectar colas pesadas y riesgo de evento.

---

## 📂 Estructura del Proyecto

```text
quant-portfolio-advisor/
│
├── app/
│   └── dashboard.py          # Interfaz interactiva en Streamlit
│
├── src/
│   ├── data/                 # Módulos de descarga y preparación de precios
│   │   └── market_data.py
│   │
│   ├── analytics/            # Motor de análisis financiero y riesgo
│   │   ├── returns.py
│   │   ├── risk_metrics.py
│   │   └── portfolio_optimizer.py
│   │
│   ├── alerts/               # Motor de reglas y análisis con IA
│   │   ├── rules_engine.py
│   │   └── ai_news.py
│   │
│   └── database/             # Gestión de la base de datos de posiciones
│       ├── db_manager.py
│       └── schema.sql
│
├── main.py                   # Script orquestador principal
├── requirements.txt          # Dependencias del proyecto
└── README.md
