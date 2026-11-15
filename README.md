# Global Solution — Segurança pública inteligente em SP (MLP)

Classificador MLP que, dadas as características de uma ocorrência **já registrada ou hipotética**
na capital paulista, estima se ela é mais compatível com **roubo** ou **furto**.
**Não** prevê se um crime vai acontecer em determinado local/horário.

- Link da aplicação web: <https://gs-aprendizado-profundo-com-redes-neurais-ccps2.streamlit.app/>
- Fonte dos dados: SSP-SP, Portal de Estatísticas — <https://www.ssp.sp.gov.br/estatistica/consultas> (seção "Bases de Dados")
- Base/período usados: Celulares subtraídos no período de 2024 

## Estrutura

```
app.py                 # aplicação web (Streamlit)
src/features.py        # atributos compartilhados entre treino e app
src/data_prep.py       # leitura, filtro da capital, rótulo, deduplicação, atributos
src/train.py           # treina 3 MLPs, compara, avalia no teste, salva artefatos
src/make_synthetic.py  # dados SINTÉTICOS só para testar o pipeline
notebook_gs_roubos.ipynb # notebook documentado (entrega 7.2)
models/                # mlp_best.keras, preprocessor.joblib, metadata.json
reports/               # curvas, matriz de confusão, tabela comparativa
data/raw/              # coloque aqui o arquivo da SSP (não versionar)
```

## Como obter os dados

1. Acesse a página da SSP-SP acima, seção **Bases de Dados**, e baixe uma base (xlsx) do ano escolhido.
2. Salve em `data/raw/` (utilizamos apenas a sheet com os dados crus).

Colunas usadas: `CIDADE` (local do **fato**, não `NOME_MUNICIPIO`, que é o da delegacia), `BAIRRO`, `DATA_OCORRENCIA_BO`,
`DESCR_SUBTIPOLOCAL`, `HORA_OCORRENCIA`/`DESCR_PERIODO`, `RUBRICA` (só para criar o rótulo; **nunca** entra como entrada da rede).

## Executar

```bash
pip install -r requirements.txt
python src/train.py          # treina e salva models/ e reports/
streamlit run app.py         # abre a aplicação
```

Rode os comandos a partir da raiz do projeto.

## Decisões principais

- **Recorte:** apenas fatos ocorridos em São Paulo (capital), pelo campo de município do fato.
- **Rótulo:** rubrica contendo "roubo" → 1; contendo "furto" → 0; ambíguos/outros crimes descartados.
- **Duplicidades:** um BO com vários itens vira uma linha.
- **Atributos:** bairro (top 60 + "OUTROS"), tipo de local (top 15 + "OUTROS"), dia da semana e período do dia (calculado pela hora; se a hora falta, usa `DESCR_PERIODO`).
- **Anti-vazamento:** rubrica/natureza fora das entradas; encoder e scaler ajustados só no treino.
- **Split:** 70/15/15 estratificado; arquitetura escolhida pela **validação (F1)**; teste usado uma vez no fim.
- **Desbalanceamento:** `class_weight` balanceado; métricas além da acurácia.
- **Arquiteturas:** A (16), B (64-32 + dropout), C (128-64-32 + dropout + L2); Adam, binary crossentropy, early stopping.

## Uso de IA generativa
- Ajudou a resolver erros de instalação no Windows e a leitura da planilha grande da SSP. Assim como em erros de Sintaxe 
- A compreender a abrangência da GS, nos apoiando para que não tenha nenhuma informação ou furo de informações visíveis por nós;
- Nos apoiou em checar todas as informações na documentação e README no fim do projeto.


## Integrantes 

- Emanuel Nabarrete RM 566931
- Luiz Eduardo RM 567417
- Eduardo Luis RM 567662
