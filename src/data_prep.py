"""Carga e limpeza da base da SSP-SP (ou de qualquer base com colunas equivalentes).

Fluxo:
  1. lê CSV/XLSX de data/raw/ (todas as abas que tenham as colunas esperadas)
  2. filtra a CAPITAL pelo campo do LOCAL DO FATO (CIDADE), não o da delegacia
  3. cria o rótulo roubo(1)/furto(0) a partir da RUBRICA e descarta o resto
  4. remove duplicidades de boletim (BO com vários itens)
  5. deriva atributos: dia da semana, período do dia, bairro e tipo de local (agrupados)
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

from features import DIAS, OUTROS, hora_para_periodo

# Nomes possíveis de cada coluna nas bases da SSP (ajuste aqui se a sua base for diferente)
ALIASES = {
    # CIDADE = município do FATO. NÃO usar NOME_MUNICIPIO: é o município da delegacia (local do registro).
    "cidade": ["CIDADE"],
    "bairro": ["BAIRRO"],
    "data": ["DATA_OCORRENCIA_BO", "DATA_OCORRENCIA"],
    "hora": ["HORA_OCORRENCIA_BO", "HORA_OCORRENCIA"],          # opcional (muitos BOs não têm)
    "periodo": ["DESCR_PERIODO", "DESC_PERIODO"],               # opcional (usado quando falta a hora)
    "local": ["DESCR_SUBTIPOLOCAL", "DESCR_TIPOLOCAL"],
    "rubrica": ["RUBRICA", "NATUREZA_APURADA"],                # usada SÓ para criar o rótulo
    "num_bo": ["NUM_BO"],
    "ano_bo": ["ANO_BO"],
    "delegacia": ["NOME_DELEGACIA"],
}
OBRIGATORIAS = ["cidade", "bairro", "data", "local", "rubrica"]  # + (hora ou periodo)
# a SSP grafa a capital de formas diferentes ("S.PAULO", "SÃO PAULO"...), já normalizadas por _norm
CAPITAL = {"SAO PAULO", "S.PAULO", "S. PAULO", "S PAULO"}
PERIODOS_SSP = {"DE MADRUGADA": "De madrugada", "PELA MANHA": "Pela manhã",
                "A TARDE": "A tarde", "A NOITE": "A noite"}  # "Em hora incerta" é descartado

_TODOS_ALIASES = {a for nomes in ALIASES.values() for a in nomes}


def _norm(txt):
    """Maiúsculo, sem acento e sem espaços sobrando."""
    if pd.isna(txt):
        return ""
    txt = unicodedata.normalize("NFKD", str(txt))
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", txt).strip().upper()


def _chave_coluna(c):
    return _norm(c).replace(" ", "_")


def _resolve_colunas(df):
    cols_norm = {_chave_coluna(c): c for c in df.columns}
    achadas = {}
    for chave, nomes in ALIASES.items():
        for n in nomes:
            if n in cols_norm:
                achadas[chave] = cols_norm[n]
                break
    return achadas


def _tem_o_necessario(mapa):
    return set(OBRIGATORIAS) <= set(mapa) and ("hora" in mapa or "periodo" in mapa)


def _abre_excel(path):
    # calamine (Rust) lê xlsx grandes em segundos; o openpyxl pode levar muitos minutos
    try:
        return pd.ExcelFile(path, engine="calamine")
    except ImportError:
        print("  (dica: pip install python-calamine acelera muito a leitura de xlsx grandes)")
        return pd.ExcelFile(path)


def _le_arquivo(path):
    path = Path(path)
    if path.suffix.lower() in (".csv", ".txt"):
        for enc in ("utf-8", "latin-1"):
            try:
                return [pd.read_csv(path, sep=None, engine="python", encoding=enc,
                                    usecols=lambda c: _chave_coluna(c) in _TODOS_ALIASES)]
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Não consegui ler {path}")
    xl = _abre_excel(path)
    abas = []
    for aba in xl.sheet_names:
        # lê só a primeira linha para saber se a aba tem as colunas certas
        # (abas como METODOLOGIA e DICIONÁRIO DE DADOS são ignoradas)
        cab = xl.parse(aba, nrows=1)
        if _tem_o_necessario(_resolve_colunas(cab)):
            abas.append(xl.parse(aba, usecols=lambda c: _chave_coluna(c) in _TODOS_ALIASES))
    return abas


def carrega_bruto(pasta="data/raw"):
    arquivos = sorted(p for p in Path(pasta).glob("*")
                      if p.suffix.lower() in (".csv", ".xlsx", ".xls", ".txt"))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum arquivo em {pasta}/. Baixe a base da SSP-SP (ver README) "
            "ou rode `python src/make_synthetic.py` para gerar dados de teste.")
    partes = []
    for p in arquivos:
        print(f"  lendo {p.name} ...")
        for d in _le_arquivo(p):
            mapa = _resolve_colunas(d)
            d = d.rename(columns={v: k for k, v in mapa.items()})
            partes.append(d[[k for k in ALIASES if k in d.columns]])
    if not partes:
        raise ValueError(
            "Nenhuma aba com as colunas esperadas (CIDADE, BAIRRO, DATA_OCORRENCIA_BO, "
            "HORA_OCORRENCIA ou DESCR_PERIODO, DESCR_SUBTIPOLOCAL/DESCR_TIPOLOCAL, RUBRICA). "
            "Atenção: NOME_MUNICIPIO é o município da delegacia, não do fato; precisa existir a "
            "coluna CIDADE. Ajuste ALIASES em data_prep.py se necessário.")
    return pd.concat(partes, ignore_index=True)


def _rotulo(rubrica):
    r = _norm(rubrica)
    tem_roubo, tem_furto = "ROUBO" in r, "FURTO" in r
    if tem_roubo and not tem_furto:
        return 1
    if tem_furto and not tem_roubo:
        return 0
    return None  # ambíguo ou outro tipo de crime -> descartado


def _parse_hora(v):
    if pd.isna(v):
        return None
    if hasattr(v, "hour"):
        return v.hour
    m = re.match(r"^\s*(\d{1,2})", str(v))
    if m and 0 <= int(m.group(1)) <= 23:
        return int(m.group(1))
    return None


def _periodo(row_hora, row_periodo):
    """Período do dia: pela hora, se existir; senão pela coluna DESCR_PERIODO da SSP."""
    h = _parse_hora(row_hora)
    if h is not None:
        return hora_para_periodo(h)
    return PERIODOS_SSP.get(_norm(row_periodo))  # None se 'Em hora incerta' ou vazio


def prepara(df, top_bairros=60, top_locais=15, n_amostra=20000, seed=42, log=print):
    """Devolve (df_modelo, info). df_modelo tem FEATURE_COLS + 'y'."""
    info = {"registros_brutos": len(df)}
    df = df.copy()
    for c in ("hora", "periodo"):
        if c not in df.columns:
            df[c] = None

    # 1) recorte geográfico: município do FATO = São Paulo
    df = df[df["cidade"].map(_norm).isin(CAPITAL)].copy()
    info["apos_filtro_capital"] = len(df)

    # 2) rótulo
    df["y"] = df["rubrica"].map(_rotulo)
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)
    info["apos_rotulo"] = len(df)

    # 3) duplicidades: um BO com vários itens vira uma única linha
    chave = [c for c in ["ano_bo", "num_bo", "delegacia"] if c in df.columns]
    if len(chave) >= 2:
        df = df.drop_duplicates(subset=chave + ["y"])
    info["apos_dedup"] = len(df)

    # 4) atributos
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["PERIODO"] = [_periodo(h, p) for h, p in zip(df["hora"], df["periodo"])]
    df = df.dropna(subset=["data", "PERIODO", "bairro", "local"])
    df["bairro"] = df["bairro"].map(_norm)
    df["local"] = df["local"].map(_norm)
    df = df[(df["bairro"] != "") & (df["local"] != "")]
    df["DIA_SEMANA"] = df["data"].dt.dayofweek.map(lambda i: DIAS[i])

    # agrupa categorias raras (reduz dimensionalidade do one-hot)
    topb = df["bairro"].value_counts().head(top_bairros).index
    topl = df["local"].value_counts().head(top_locais).index
    df["BAIRRO"] = df["bairro"].where(df["bairro"].isin(topb), OUTROS)
    df["LOCAL"] = df["local"].where(df["local"].isin(topl), OUTROS)
    info["apos_limpeza"] = len(df)

    # 5) amostra estratificada (mantém a proporção original roubo/furto)
    if n_amostra and len(df) > n_amostra:
        df = df.groupby("y").sample(frac=n_amostra / len(df), random_state=seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    info["registros_modelo"] = len(df)
    info["proporcao_roubo"] = round(float(df["y"].mean()), 4)

    for k, v in info.items():
        log(f"  {k}: {v}")
    cols = ["BAIRRO", "LOCAL", "DIA_SEMANA", "PERIODO", "y"]
    return df[cols], info
