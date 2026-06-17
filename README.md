# Predição de Vendas com Séries Temporais utilizando PyTorch

Projeto desenvolvido para a disciplina **Engenharia de Software para IA e Frameworks Profundos** do programa de pós-graduação em **Deep Learning** do **Centro de Informática (CIn) da Universidade Federal de Pernambuco (UFPE)**.

## Objetivo

O objetivo deste projeto é desenvolver um modelo de previsão de vendas através de técnicas de **Séries Temporais**, utilizando **PyTorch**, aplicadas ao conjunto de dados da Walmart.

## Problema de Negócio

A previsão de vendas é um desafio importante no varejo, pois permite antecipar demandas futuras e apoiar o planejamento operacional.

Neste projeto, buscamos explorar os padrões temporais presentes no histórico de vendas da Walmart para construir um modelo capaz de realizar previsões futuras com base em dados históricos.

## Dataset

O projeto utiliza o dataset de vendas da Walmart.

Os arquivos de dados devem ser armazenados na pasta:

```text
data/
```

## Estrutura do Projeto

```text
pred-demanda-2026-1
│
├── data/
│   └── *.csv
│
├── src/
│   │
│   ├── data/
│   │   └── data_loader.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   ├── models/
│   │   └── model.py
│   │
│   ├── preprocessing/
│   │   ├── features.py
│   │   └── transform.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   ├── utils/
│   │   └── config.py
│   │
│   └── main.py
│
└── README.md
```

Programa de Pós-Graduação em Deep Learning

Centro de Informática (CIn) – Universidade Federal de Pernambuco (UFPE)
