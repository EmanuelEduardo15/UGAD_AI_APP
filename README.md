"""
# UGAD+ AI App

Aplicativo para simulação da fidelidade de estados quânticos com base no sistema UGAD+, utilizando regressão IA com Streamlit.

## 🚀 Instalação e Execução Local

1. Crie e ative um ambiente virtual (opcional mas recomendado):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
 
2. Instale as dependências:
```bash
pip install streamlit scikit-learn matplotlib pandas
```
3.Execute o aplicativo:
```bash
 streamlit run ugad_ai.py
# 🖥️ Executável (.exe) para Windows
```
1.	Instale o cx_Freeze:
   ```bash
pip install cx_Freeze
```
2.	Gere o executável:
```
python ugad_ai.py --build-exe
###O executável e arquivos necessários serão criados em build/.
```
# ☁️ Implantação na Web
1.	Envie este arquivo a um repositório GitHub público.
	2.	Acesse https://streamlit.io/cloud e conecte seu GitHub.
	3.	Clique em New app, selecione o repositório e o arquivo ugad_ai.py, depois Deploy.


Desenvolvido por Emanuel Eduardo
ChQuantum Technologies
“””

import sys
import os
import argparse
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

————————————————

# Interface Streamlit / Main App

————————————————

def run_app():
st.set_page_config(page_title=“UGAD+ Fidelity AI”, layout=“centered”)
st.title(“UGAD+ - Previsão de Fidelidade Quântica com IA”)
st.markdown(“Simulação e ajuste baseado nos dados experimentais do sistema UGAD+.”)
# Dados simulados (substitua pelos reais)
dados = {
    "f": [0.7, 0.8, 0.9, 1.0],
    "fidelidade": [99.05, 99.49, 99.74, 99.80]
}
df = pd.DataFrame(dados)

# Treinamento de modelos
X = df["f"].values.reshape(-1, 1)
y = df["fidelidade"].values

lin = LinearRegression().fit(X, y)
poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X, y)

# Gráfico de ajuste
fig, ax = plt.subplots()
ax.scatter(X, y, color='black', label="Dados experimentais")
f_range = np.linspace(0.6, 1.2, 100).reshape(-1,1)
ax.plot(f_range, lin.predict(f_range), '--', label="Linear")
ax.plot(f_range, poly.predict(f_range), '-', label="Polinomial (grau 2)")
ax.set_xlabel("Parâmetro f")
ax.set_ylabel("Fidelidade (%)")
ax.set_title("Ajustes de Modelos")
ax.legend()
st.pyplot(fig)

# Erros dos modelos
st.subheader("Desempenho dos Modelos")
st.write(f"- RMSE Linear: {mean_squared_error(y, lin.predict(X), squared=False):.4f}")
st.write(f"- RMSE Polinomial (g2): {mean_squared_error(y, poly.predict(X), squared=False):.4f}")

# Previsão interativa
st.subheader("Prever Fidelidade para novo f")
f_input = st.slider("Escolha o valor de f", 0.6, 1.2, 1.0, 0.01)
fid_pred = poly.predict(np.array([[f_input]]))[0]
st.write(f"**Previsão: fidelidade = {fid_pred:.2f}% para f = {f_input:.2f}**")

# Mostrar dados
st.subheader("Dados Utilizados")
st.dataframe(df)
————————————————

# Geração de Executável via cx_Freeze

————————————————

def build_executable():
from cx_Freeze import setup, Executable
setup(
name=“UGAD+ AI”,
version=“1.0”,
description=“Aplicativo UGAD+ com IA”,
executables=[Executable(“ugad_ai.py”)]
)
print(“Executável gerado em build/”)

————————————————

# Impressão do README em Markdown

————————————————

def print_readme():
print(doc)

————————————————

# Entrada principal

————————————————

if name == “main”:
parser = argparse.ArgumentParser(description=“UGAD+ AI App”)
parser.add_argument(
“–print-readme”, action=“store_true”,
help=“Exibe o README em Markdown”
)
parser.add_argument(
“–build-exe”, action=“store_true”,
help=“Gera executável usando cx_Freeze”
)
args = parser.parse_args()
if args.print_readme:
    print_readme()
    sys.exit(0)
if args.build_exe:
    build_executable()
    sys.exit(0)

# Se iniciado via `streamlit run ugad_ai.py`, executa o app
run_app()

**Como usar este único arquivo:**

- **Para rodar o app:**
  ```bash
  streamlit run ugad_ai.py
  ```
 ## •	Para ver o README em Markdown:
 python ugad_ai.py --print-readme

## •	Para criar o executável (.exe):
python ugad_ai.py --build-exe
