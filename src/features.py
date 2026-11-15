"""Atributos compartilhados entre treino e aplicação web.

Usar o MESMO código nos dois lados evita "training-serving skew":
o app monta exatamente as mesmas colunas que a rede viu no treino.

Atributos de entrada (todos categóricos, codificados com one-hot):
  BAIRRO, LOCAL (tipo de local), DIA_SEMANA, PERIODO (madrugada/manhã/tarde/noite).
"""
import pandas as pd

CAT_COLS = ["BAIRRO", "LOCAL", "DIA_SEMANA", "PERIODO"]
NUM_COLS = []  # sem atributos numéricos: a hora exata falta em boa parte dos BOs da SSP
FEATURE_COLS = CAT_COLS + NUM_COLS

DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
        "sexta-feira", "sábado", "domingo"]
PERIODOS = ["De madrugada", "Pela manhã", "A tarde", "A noite"]
OUTROS = "OUTROS"


def hora_para_periodo(hora):
    h = int(hora)
    if h < 6:
        return "De madrugada"
    if h < 12:
        return "Pela manhã"
    if h < 18:
        return "A tarde"
    return "A noite"


def monta_linha(bairro, local, dia_semana, periodo):
    """Cria o DataFrame de 1 linha usado pelo app web na inferência."""
    return pd.DataFrame([{
        "BAIRRO": bairro, "LOCAL": local, "DIA_SEMANA": dia_semana, "PERIODO": periodo,
    }])[FEATURE_COLS]
