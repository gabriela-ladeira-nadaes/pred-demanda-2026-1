# Previsão de Vendas com Séries Temporais

Projeto desenvolvido para a disciplina **Engenharia de Software para IA e Frameworks Profundos** do programa de pós-graduação em Deep Learning do Centro de Informática (CIn) da Universidade Federal de Pernambuco (UFPE).

## Objetivo

Desenvolver e avaliar um modelo de previsão de vendas por meio de técnicas de Séries Temporais, utilizando PyTorch.

## Problema de Negócio

A previsão de vendas é um desafio importante no varejo, pois permite antecipar demandas futuras e apoiar o planejamento operacional. Este projeto explora os padrões temporais presentes no histórico de vendas do Walmart para construir um modelo capaz de realizar previsões futuras a partir de dados históricos.

## Dataset

O projeto utiliza o dataset de vendas do Walmart (`walmart_dataset_sales.csv`, versionado em `data/`), com 421.570 registros semanais entre 2010-02-05 e 2012-10-26, cobrindo 45 lojas e 81 departamentos. A variável alvo é `Weekly_Sales`. Além de atributos contínuos (temperatura, preço de combustível, CPI, desemprego, markdowns, tamanho da loja), há identificadores já codificados como inteiros (`Store`, `Dept`, `Type`, `IsHoliday`).

## Estrutura do Projeto

```text
pred-demanda-2026-1
│
├── data/
│   └── walmart_dataset_sales.csv
│
├── src/
│   ├── data/
│   │   ├── data_loader.py        # carregamento e validação dos dados
│   │   └── datasets.py           # tensores, TensorDataset e DataLoader
│   ├── preprocessing/
│   │   ├── transform.py          # limpeza, split cronológico e normalização (NumPy)
│   │   └── features.py           # criação de atributos temporais
│   ├── models/
│   │   └── model.py              # esqueleto do modelo (Etapa 6)
│   ├── training/
│   │   └── train.py              # esqueleto do treinamento (Etapa 6)
│   ├── evaluation/
│   │   └── metrics.py            # esqueleto da avaliação (Etapa 6/7)
│   ├── utils/
│   │   └── config.py             # configurações, caminhos e colunas
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

---

## Entrega 1 — Etapas 1 a 3 (Fundamentos de software)

- **Etapa 1 — Funções e repositório:** repositório criado no GitHub, funções iniciais (`load_data`, `clean_data`, `split_data`, `main`), primeiro script executável e documentação inicial.
- **Etapa 2 — Modularização:** código separado em pacotes (`data`, `preprocessing`, `models`, `training`, `evaluation`, `utils`), com `__init__.py` em cada um e execução centralizada em `main.py`.
- **Etapa 3 — Tipagem:** funções principais com *type hints* explícitos de entrada e saída, tornando os contratos de cada função claros.

---

## Entrega 2 — Etapas 4 e 5 (NumPy e PyTorch - Parte 1) e motivações

### Etapa 4 — Pré-processamento com NumPy (`preprocessing/transform.py`)

**Ordenação cronológica:** `clean_data` ordena o dataset por data (`sort_data`) e reindexa. Isso é pré-requisito para o split temporal, pois o corte é feito por posição de linha, então as linhas precisam estar em ordem de tempo.

**Matriz de features com ordem fixa (`build_features_matrix`):** Primeiro são adicionadas as colunas contínuas e depois as categóricas, sempre nessa ordem para posteriormente selecionar o bloco de variáveis contínuas com um único *slice* (`X[:, :n_continuous]`), sem precisar rastrear índices de colunas espalhados.

**Split cronológico, não aleatório (`split_data`):** Como o problema é de séries temporais, o conjunto de teste representa o futuro: os primeiros 80% das linhas vão para treino e os últimos 20% para teste. Um split aleatório misturaria passado e futuro, vazando informação.

**Padronização (z-score) apenas no bloco contínuo (`standardize`):** Só as colunas contínuas são padronizadas. As categóricas (`Store`, `Dept`, `Type`, `IsHoliday`) são identificadores codificados como inteiros. Média e desvio padrão das colunas contínuas são estimados somente no treino, pois estimar essas estatísticas usando o teste levaria a um vazamento de informação já que a transformação carregaria informação do futuro. 

**Atributos temporais (`features.py`):** `create_features` obtém `Year`, `Month` e `WeekOfYear` da data. A motivação é codificar sazonalidade e tendência como features numéricas tal que cada linha vira um exemplo independente.

### Etapa 5 — Introdução ao PyTorch / Parte 1 (`data/datasets.py`)

**Conversão para tensores `float32` (`to_tensors`):** Os arrays NumPy viram `torch.Tensor` em `float32`, o tipo padrão das operações de rede neural. Aplicamos reshape sobre o alvo `y` para `(N, 1)` porque as funções de perda de regressão do PyTorch esperam que predição e alvo tenham o mesmo formato `(N, 1)`.

**`TensorDataset` + `DataLoader` (`make_dataset`, `make_dataloader`):** O `TensorDataset` armazena `X` e `y`, e o `DataLoader` cria os batches.

## Status

Pipeline das Etapas 1 a 5 implementado com carregamento, limpeza, pré-processamento com NumPy e geração dos `DataLoaders` de treino e teste. A próxima etapa será a implementação do modelo.

---

Programa de Pós-Graduação em Deep Learning

Centro de Informática (CIn) – Universidade Federal de Pernambuco (UFPE)