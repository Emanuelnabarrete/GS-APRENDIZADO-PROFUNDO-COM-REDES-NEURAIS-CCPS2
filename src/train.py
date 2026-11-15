"""Treina 3 MLPs, compara na VALIDAÇÃO, avalia o escolhido no TESTE e salva artefatos.

Uso:  python src/train.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from data_prep import carrega_bruto, prepara
from features import CAT_COLS, NUM_COLS, FEATURE_COLS

SEED = 42
BATCH = 64
EPOCHS = 60
OUT_MODELS, OUT_REPORTS = Path("models"), Path("reports")


# ------------------------------------------------------------------ arquiteturas
def build_mlp(n_in, hidden, activation="relu", dropout=0.0, l2=0.0, lr=1e-3):
    """MLP binária: entrada -> [Dense(+Dropout)]* -> Dense(1, sigmoid)."""
    reg = regularizers.l2(l2) if l2 else None
    m = keras.Sequential([layers.Input(shape=(n_in,))])
    for n in hidden:
        m.add(layers.Dense(n, activation=activation, kernel_regularizer=reg))
        if dropout:
            m.add(layers.Dropout(dropout))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=["accuracy"])
    return m


ARQUITETURAS = {
    "A_simples":   dict(hidden=[16], activation="relu", dropout=0.0, l2=0.0, lr=1e-3),
    "B_media":     dict(hidden=[64, 32], activation="relu", dropout=0.3, l2=0.0, lr=1e-3),
    "C_profunda":  dict(hidden=[128, 64, 32], activation="relu", dropout=0.3, l2=1e-4, lr=5e-4),
}


def metricas(y, prob, thr=0.5):
    p = (prob >= thr).astype(int)
    return {"acuracia": accuracy_score(y, p),
            "precision": precision_score(y, p, zero_division=0),
            "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0)}


def main():
    for d in (OUT_MODELS, OUT_REPORTS):
        d.mkdir(exist_ok=True)
    keras.utils.set_random_seed(SEED)

    print("== Preparação dos dados")
    bruto = carrega_bruto("data/raw")
    df, info = prepara(bruto, seed=SEED)

    # ---- split 70/15/15 estratificado (teste intocado até o fim)
    X, y = df[FEATURE_COLS], df["y"].values
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.15 / 0.85,
                                              stratify=y_tmp, random_state=SEED)

    # ---- pré-processador ajustado SÓ no treino (evita vazamento)
    transformadores = [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)]
    if NUM_COLS:
        transformadores.append(("num", StandardScaler(), NUM_COLS))
    pre = ColumnTransformer(transformadores)
    A_tr = pre.fit_transform(X_tr).astype("float32")
    A_va, A_te = pre.transform(X_va).astype("float32"), pre.transform(X_te).astype("float32")
    n_in = A_tr.shape[1]
    print(f"  treino={len(A_tr)} validação={len(A_va)} teste={len(A_te)} | entradas={n_in}")

    # pesos de classe: compensa desbalanceamento roubo x furto
    pos = y_tr.mean()
    cw = {0: 0.5 / (1 - pos), 1: 0.5 / pos}

    # ---- experimentos
    print("== Treinando arquiteturas")
    resultados, historicos, modelos = [], {}, {}
    for nome, cfg in ARQUITETURAS.items():
        keras.utils.set_random_seed(SEED)
        m = build_mlp(n_in, **cfg)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        h = m.fit(A_tr, y_tr, validation_data=(A_va, y_va), epochs=EPOCHS, batch_size=BATCH,
                  class_weight=cw, callbacks=[es], verbose=0)
        mv = metricas(y_va, m.predict(A_va, verbose=0).ravel())
        mt = metricas(y_tr, m.predict(A_tr, verbose=0).ravel())
        resultados.append({"modelo": nome, "parametros": int(m.count_params()),
                           "epocas": len(h.history["loss"]),
                           **{f"val_{k}": v for k, v in mv.items()},
                           "treino_acuracia": mt["acuracia"]})
        historicos[nome], modelos[nome] = h.history, m
        print(f"  {nome}: val_acc={mv['acuracia']:.3f} val_f1={mv['f1']:.3f} "
              f"(treino_acc={mt['acuracia']:.3f}, épocas={len(h.history['loss'])})")

    tab = pd.DataFrame(resultados).round(4)
    tab["gap_treino_val"] = (tab["treino_acuracia"] - tab["val_acuracia"]).round(4)
    tab.to_csv(OUT_REPORTS / "comparacao_arquiteturas.csv", index=False)
    print(tab.to_string(index=False))

    # ---- seleção PELA VALIDAÇÃO (F1)
    melhor = tab.sort_values("val_f1", ascending=False).iloc[0]["modelo"]
    print(f"== Modelo escolhido (maior F1 na validação): {melhor}")
    m = modelos[melhor]

    # ---- avaliação final no TESTE (uma única vez)
    prob_te = m.predict(A_te, verbose=0).ravel()
    mt = metricas(y_te, prob_te)
    print("== TESTE:", {k: round(v, 4) for k, v in mt.items()})

    # ---- gráficos
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    h = historicos[melhor]
    ax[0].plot(h["accuracy"], label="treino"); ax[0].plot(h["val_accuracy"], label="validação")
    ax[0].set_title(f"Acurácia - {melhor}"); ax[0].set_xlabel("época"); ax[0].legend()
    ax[1].plot(h["loss"], label="treino"); ax[1].plot(h["val_loss"], label="validação")
    ax[1].set_title(f"Perda - {melhor}"); ax[1].set_xlabel("época"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT_REPORTS / "curvas_modelo_escolhido.png", dpi=150); plt.close(fig)

    cm = confusion_matrix(y_te, (prob_te >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["furto", "roubo"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusão (teste)")
    fig.tight_layout(); fig.savefig(OUT_REPORTS / "matriz_confusao.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for nome, hh in historicos.items():
        ax.plot(hh["val_loss"], label=nome)
    ax.set_title("Perda de validação por arquitetura"); ax.set_xlabel("época"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT_REPORTS / "val_loss_comparacao.png", dpi=150); plt.close(fig)

    # ---- artefatos para o app
    m.save(OUT_MODELS / "mlp_best.keras")
    joblib.dump(pre, OUT_MODELS / "preprocessor.joblib")
    cat_enc = pre.named_transformers_["cat"]
    meta = {
        "modelo": melhor, "config": ARQUITETURAS[melhor],
        "opcoes": {c: sorted(map(str, cats)) for c, cats in zip(CAT_COLS, cat_enc.categories_)},
        "classes": {"0": "furto", "1": "roubo"},
        "metricas_teste": {k: round(float(v), 4) for k, v in mt.items()},
        "info_dados": info,
        "tamanhos": {"treino": len(A_tr), "validacao": len(A_va), "teste": len(A_te)},
    }
    (OUT_MODELS / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("== Artefatos salvos em models/ e reports/")


if __name__ == "__main__":
    main()
