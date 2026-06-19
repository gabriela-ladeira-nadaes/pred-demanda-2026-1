# Previsão de Vendas com Séries Temporais 

Projeto desenvolvido para a disciplina **Engenharia de Software para IA e Frameworks Profundos** do programa de pós-graduação em **Deep Learning** do **Centro de Informática (CIn) da Universidade Federal de Pernambuco (UFPE)**.

## Objetivo

Desenvolver e avaliar um modelo de previsão de vendas por meio de técnicas de **Séries Temporais**, utilizando o framework PyTorch, com base no histórico de vendas do Walmart.

## Problema de Negócio

A previsão de vendas é um desafio importante no varejo, pois permite antecipar demandas futuras e apoiar o planejamento operacional.

Este projeto busca explorar os padrões temporais presentes no histórico de vendas do Walmart para construir um modelo capaz de realizar previsões futuras com base em dados históricos.

## Dataset

O projeto utiliza o dataset de vendas da Walmart. O arquivo `walmart_dataset_sales.csv` está versionado na pasta:

```text
data/
```

## Estrutura Proposta

```text
pred-demanda-2026-1
│
├── data/
│   └── walmart_dataset_sales.csv
│
├── src/
│   ├── data/
│   │   └── data_loader.py        # carregamento e validação dos dados
│   ├── preprocessing/
│   │   ├── transform.py          # limpeza e separação treino/teste
│   │   └── features.py           # criação de features
│   ├── models/
│   │   └── model.py              # definição/salvamento do modelo
│   ├── training/
│   │   └── train.py              # treinamento do modelo
│   ├── evaluation/
│   │   └── metrics.py            # avaliação por métricas
│   ├── utils/
│   │   └── config.py             # configurações e caminhos
│   └── main.py                   # ponto de entrada (orquestra o pipeline)
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Como executar

A partir da raiz do projeto:

```bash
pip install -r requirements.txt
python src/main.py
```

O caminho do dataset é resolvido em `src/utils/config.py` de forma independente do diretório de execução, evitando erros de caminho relativo.

## Entrega 1 (Tópicos 1 a 3) 

Esta primeira entrega cobre as três primeiras etapas da disciplina. As funções de pré-processamento, treinamento e avaliação estão como esqueletos tipados, a serem implementados nas próximas etapas.

**Etapa 1 — Funções, modularização inicial e repositório**
- Repositório criado no GitHub com os arquivos iniciais.
- Funções iniciais do sistema definidas (`load_data`, `clean_data`, `split_data`, `main`).
- Primeiro script executável (`src/main.py`) rodando de ponta a ponta.
- `README.md` com descrição do problema, dataset e instruções de execução.
- `requirements.txt` com as dependências do projeto.

**Etapa 2 — Modularização**
- Código dividido em módulos e pacotes (`data`, `preprocessing`, `models`, `training`, `evaluation`, `utils`).
- Imports funcionando corretamente, com `__init__.py` em cada pacote.
- Separação clara entre dados, modelo, treinamento, avaliação e utilidades.
- Execução centralizada em `main.py`, que orquestra todo o pipeline.

**Etapa 3 — Tipagem**
- Funções principais com *type hints* (tipos de entrada e saída explícitos).
- Contratos de função claros, reduzindo ambiguidade.
- Tipos coerentes com o que cada função efetivamente recebe e retorna.

## Status

Estrutura inicial modularizada e tipada (Etapas 1–3). As próximas etapas implementarão o pré-processamento com NumPy, o modelo em PyTorch, os experimentos e os testes automatizados.

---

Programa de Pós-Graduação em Deep Learning

Centro de Informática (CIn) – Universidade Federal de Pernambuco (UFPE)
