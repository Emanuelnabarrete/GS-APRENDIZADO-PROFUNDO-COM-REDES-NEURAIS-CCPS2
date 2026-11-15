"""Aplicação web (Streamlit) para testar a MLP treinada.

Rodar:  streamlit run app.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).parent / "src"))

import joblib
import streamlit as st
from tensorflow import keras

from features import DIAS, PERIODOS, monta_linha

BASE = Path(__file__).parent
st.set_page_config(page_title="Roubo ou furto? - MLP SP", page_icon="🔎")


@st.cache_resource
def carrega():
    modelo = keras.models.load_model(BASE / "models" / "mlp_best.keras")
    pre = joblib.load(BASE / "models" / "preprocessor.joblib")
    meta = json.loads((BASE / "models" / "metadata.json").read_text())
    return modelo, pre, meta


st.title("Classificação de ocorrências em São Paulo")
st.caption("MLP que estima se um registro com estas características é mais compatível "
           "com **roubo** ou **furto**.")

try:
    modelo, pre, meta = carrega()
except Exception as e:  # modelo ainda não treinado
    st.error("Modelo não encontrado. Rode `python src/train.py` primeiro.")
    st.exception(e)
    st.stop()

op = meta["opcoes"]
with st.form("form"):
    c1, c2 = st.columns(2)
    bairro = c1.selectbox("Bairro", op["BAIRRO"])
    local = c2.selectbox("Tipo de local", op["LOCAL"])
    dia = c1.selectbox("Dia da semana", DIAS, index=4)
    periodo = c2.selectbox("Período do dia", PERIODOS, index=3)
    enviar = st.form_submit_button("Classificar")

if enviar:
    linha = monta_linha(bairro, local, dia, periodo)
    x = pre.transform(linha).astype("float32")
    p_roubo = float(modelo.predict(x, verbose=0)[0, 0])
    p_furto = 1 - p_roubo
    classe = "ROUBO" if p_roubo >= 0.5 else "FURTO"

    st.subheader(f"Classe estimada: {classe}")
    c1, c2 = st.columns(2)
    c1.metric("Roubo", f"{p_roubo:.1%}")
    c2.metric("Furto", f"{p_furto:.1%}")
    st.progress(p_roubo)
    st.info("Interpretação: **dado que** uma ocorrência aconteceu com essas características, "
            "essa é a probabilidade que o modelo atribui a cada classe. "
            "Isto **não** é o risco de um crime acontecer nesse local ou horário.")

with st.expander("Sobre o modelo"):
    st.write(f"Arquitetura escolhida: **{meta['modelo']}** — {meta['config']}")
    st.write("Métricas no conjunto de teste:", meta["metricas_teste"])
    st.write("Atributos usados: bairro, tipo de local, dia da semana e período do dia.")
    st.write("Dados:", meta["info_dados"])
