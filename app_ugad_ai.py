import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="UGAD+ Fidelity AI", layout="centered")

st.title("UGAD+ - Previsão de Fidelidade Quântica com IA")
st.markdown("Simulação e ajuste baseado nos dados experimentais do sistema UGAD+.")

# === Dados simulados (substitua pelos reais, se quiser) ===
dados = {
    "f": [0.7, 0.8, 0.9, 1.0],
    "fidelidade": [99.05, 99.49, 99.74, 99.80]
}
df = pd.DataFrame(dados)

# === Treinamento do modelo linear ===
X = df["f"].values.reshape(-1, 1)
y = df["fidelidade"].values

modelo_linear = LinearRegression()
modelo_linear.fit(X, y)
y_pred_linear = modelo_linear.predict(X)

# === Modelo polinomial para melhor ajuste ===
poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly.fit(X, y)
y_pred_poly = poly.predict(X)

# === Gráficos ===
fig, ax = plt.subplots()
ax.scatter(X, y, color='black', label="Dados experimentais")
ax.plot(X, y_pred_linear, color='blue', linestyle='--', label="Linear")
ax.plot(X, y_pred_poly, color='red', label="Polinomial (grau 2)")
ax.set_xlabel("Parâmetro f")
ax.set_ylabel("Fidelidade (%)")
ax.set_title("Ajustes de Modelos")
ax.legend()
st.pyplot(fig)

# === Erros ===
rmse_linear = mean_squared_error(y, y_pred_linear, squared=False)
rmse_poly = mean_squared_error(y, y_pred_poly, squared=False)

st.subheader("Desempenho dos Modelos")
st.write(f"**RMSE Linear:** {rmse_linear:.4f}")
st.write(f"**RMSE Polinomial (grau 2):** {rmse_poly:.4f}")

# === Previsão Interativa ===
st.subheader("Prever Fidelidade para novo f")
f_input = st.slider("Escolha o valor de f", min_value=0.6, max_value=1.2, value=1.0, step=0.01)
f_input_array = np.array([[f_input]])

fid_pred = poly.predict(f_input_array)[0]
st.write(f"**Previsão de Fidelidade para f = {f_input:.2f}: {fid_pred:.2f}%**")

# === Mostrar os dados ===
st.subheader("Dados Utilizados")
st.dataframe(df)

# Assinatura
st.markdown("---")
st.markdown("Desenvolvido por Emanuel Eduardo • ChQuantum Technologies")
