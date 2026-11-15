"""Gera uma base SINTÉTICA no formato da SSP-SP, só para testar o pipeline.

NÃO use isto como resultado final: os padrões aqui são inventados.
Troque pelo arquivo real da SSP em data/raw/ (veja README.md).
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(0)
N = 40000

bairros = ["SE", "PINHEIROS", "MOEMA", "ITAIM BIBI", "SANTANA", "TATUAPE", "PENHA", "LAPA",
           "VILA MARIANA", "SANTO AMARO", "BELA VISTA", "CONSOLACAO", "JARDIM PAULISTA",
           "BOM RETIRO", "BRAS", "MOOCA", "IPIRANGA", "CAPAO REDONDO", "GRAJAU", "ITAQUERA"]
locais = ["Via pública", "Residência", "Estabelecimento comercial", "Transporte coletivo",
          "Shopping center", "Condomínio residencial", "Estabelecimento de ensino"]
periodos = ["De madrugada", "Pela manhã", "A tarde", "A noite"]

hora = rng.integers(0, 24, N)
local = rng.choice(locais, N, p=[.45, .12, .15, .10, .06, .07, .05])
bairro = rng.choice(bairros, N)
data = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")

# probabilidade de ser roubo depende (de forma inventada) de hora e local
logit = -0.3 + 1.0 * np.isin(local, ["Via pública", "Transporte coletivo"]) \
        - 1.2 * np.isin(local, ["Residência", "Condomínio residencial", "Shopping center"]) \
        + 0.9 * ((hora >= 18) | (hora < 5)) + rng.normal(0, 0.8, N)
roubo = rng.random(N) < 1 / (1 + np.exp(-logit))
rubrica = np.where(roubo, "Roubo (art. 157)", "Furto (art. 155)")

df = pd.DataFrame({
    "ANO_BO": 2024, "NUM_BO": np.arange(N), "NOME_DELEGACIA": "01o DP",
    "CIDADE": np.where(rng.random(N) < .9, "S.PAULO", "GUARULHOS"),
    "BAIRRO": bairro, "DATA_OCORRENCIA_BO": data,
    "HORA_OCORRENCIA_BO": [f"{h:02d}:{rng.integers(0,60):02d}:00" for h in hora],
    "DESCR_SUBTIPOLOCAL": local, "RUBRICA": rubrica,
})
df["CIDADE"] = df["CIDADE"].replace("S.PAULO", "SÃO PAULO")
# alguns outros crimes e duplicatas de BO para exercitar os filtros
extra = df.sample(1500, random_state=1).copy()
extra["RUBRICA"] = "Estupro (art. 213)"
dup = df.sample(800, random_state=2).copy()
df = pd.concat([df, extra, dup], ignore_index=True)

Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/CelularesSubtraidos_2024.xlsx", index=False)
print("Gerado data/raw/CelularesSubtraidos_2024.xlsx com", len(df), "linhas (SINTÉTICO)")
