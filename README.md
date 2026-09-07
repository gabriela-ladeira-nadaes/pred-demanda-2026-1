# Predição de Demanda — Vendas Semanais do Walmart

Projeto da disciplina **Engenharia de Software para IA e Frameworks Profundos** do programa de pós-graduação em Deep Learning do Centro de Informática (CIn) da Universidade Federal de Pernambuco (UFPE).

## Problema e objetivo

As variações semanais das vendas dificultam o planejamento de abastecimento das lojas. O sistema aprende os padrões históricos de venda por loja e departamento — junto com feriados, promoções, características da loja e indicadores econômicos — para projetar as vendas das próximas semanas e apoiar a priorização do abastecimento.

O levantamento de requisitos foi feito com o framework **GR4ML** (documento `predicao_demandas_GR4ML.pdf`, na raiz do repositório), que define o WMAE como métrica principal e origina os requisitos funcionais, não funcionais e de dados.

## Dataset

`data/walmart_dataset_sales.csv` — 421.570 registros semanais entre 2010-02-05 e 2012-10-26, cobrindo 45 lojas e 81 departamentos.

| Grupo | Colunas |
| --- | --- |
| Identificadores | `Store`, `Dept`, `Type`, `Size` |
| Temporais | `Date` e, derivados dela, `Year`, `Month`, `WeekOfYear` |
| Sinalizador | `IsHoliday` |
| Econômicas | `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` |
| Promoções | `MarkDown1` … `MarkDown5` |
| **Alvo** | `Weekly_Sales` |

Os identificadores já vêm codificados como inteiros no CSV, e os markdowns já vêm com zeros no lugar dos ausentes.

## Arquitetura

O projeto tem três fluxos que compartilham os mesmos módulos:

**Treino** (`src/main.py`) — carrega e valida o CSV, limpa e deriva os atributos temporais, faz o split cronológico, ajusta o pré-processamento, monta tensores e `DataLoader`s, treina e escolhe o melhor modelo por WMAE.

**Persistência** (`models/model.py`) — O artefato são três arquivos com o mesmo timestamp: os pesos, o `StandardScaler` do alvo e o `ColumnTransformer` das features.

**Inferência** (`src/predict.py`) — localiza o artefato mais recente, projeta as próximas semanas para toda a rede ou para uma loja/departamento específico, e gera os gráficos de projeção.

```
CSV  →  data/  →  preprocessing/  →  data/datasets  →  training/  →  evaluation/
                        │                                   │
                        └──────────► models/ (artefato: pesos + scaler + transformer)
                                             │
                          dado novo ─────────┴────► inference/ ──► projeção + gráficos
```

## Estrutura do projeto

```text
pred-demanda-2026-1
│
├── data/
│   └── walmart_dataset_sales.csv
│
├── models/                        # artefatos gerados pelo treino (.pth + .pkl)
│
├── src/
│   ├── data/
│   │   ├── data_loader.py         # leitura e validação do CSV
│   │   └── datasets.py            # tensores, TensorDataset, DataLoader e device
│   ├── preprocessing/
│   │   ├── features.py            # atributos temporais derivados da data
│   │   └── transform.py           # limpeza, split cronológico e normalização
│   ├── models/
│   │   └── model.py               # arquiteturas de rede e persistência do artefato
│   ├── training/
│   │   └── train.py               # laço de treino, loss ponderada e seleção da época
│   ├── evaluation/
│   │   └── metrics.py             # WMAE
│   ├── inference/
│   │   ├── predictor.py           # carregamento do artefato e geração da projeção
│   │   └── visualization.py       # gráficos de projeção
│   ├── utils/
│   │   └── config.py              # caminhos, hiperparâmetros e listas de colunas
│   ├── main.py                    # ponto de entrada do treino
│   └── predict.py                 # ponto de entrada da inferência
│
├── tests/                         # suíte unittest
├── requirements.txt
└── README.md
```

A separação segue o ciclo de vida do dado, não as camadas técnicas: cada pasta é uma etapa do pipeline. Isso mantém cada módulo com uma responsabilidade única e permite testar o pré-processamento sem carregar PyTorch, ou trocar a arquitetura da rede sem tocar no laço de treino.

## Como executar

**A partir da raiz do repositório.** 

```bash
pip install -r requirements.txt
```

Treinar e salvar o melhor modelo:

```bash
python src/main.py
```

Gerar a projeção a partir do artefato mais recente:

```bash
python src/predict.py
```

Rodar a suíte de testes:

```bash
python -m unittest discover -s tests
```

## Pré-processamento

1. **Ordenação e limpeza** (`clean_data`) — descarta a coluna de índice do CSV, converte `Date` para datetime e ordena por `Store`, `Dept`, `Date`.
2. **Atributos temporais** (`create_features`) — extrai `Year`, `Month` e `WeekOfYear` da data, codificando sazonalidade e tendência.
3. **Split cronológico** (`split_data`) — o corte é a data de `CUTOFF_DATE`: treino é tudo antes de 2012-01-01 e teste é tudo depois.
4. **Normalização** (`standardize`) — um `ColumnTransformer` aplica, nesta ordem: one-hot em `Store`, `Dept`, `Type`, `Month` e `WeekOfYear`; z-score nas numéricas e em `Year`; e `IsHoliday` não tem  transformação.
5. **Normalização do alvo** (`standardize_y`) — `Weekly_Sales` é padronizado, e o scaler é devolvido para desfazer a transformação na hora de reportar métricas em dólares.

Tanto o `ColumnTransformer` quanto o scaler do alvo são ajustados no treino e aplicados ao teste. O teste `test_standardize_fits_only_on_train` verifica essa propriedade conferindo que a transformação do treino não muda.

> **Invariante**: `IsHoliday` é passado por último no `ColumnTransformer` e, por isso, é a última coluna da matriz de features. A loss de treino depende dessa posição (`batch_X[:, -1]`).

## Modelo e treino

`src/models/model.py` define três arquiteturas: `LinearRegression` (referência), `FinancialModel` (MLP) e `LSTMModel`. O `main.py` percorre um dicionário de modelos, treina cada um e guarda o de menor WMAE.

**`FinancialModel`** — MLP de três camadas ocultas: `Linear(input, 128) → ReLU → BatchNorm → Dropout(0.4)` → `Linear(128, 64) → ReLU → BatchNorm → Dropout(0.4)` → `Linear(64, 32) → ReLU` → `Linear(32, 1)`.

| Hiperparâmetro | Valor |
| --- | --- |
| Loss | `weighted_l1_loss` — L1 com peso 5 em semanas de feriado |
| Otimizador | Adam, `lr=0.0001`, `weight_decay=1e-4` |
| Gradient clipping | `max_norm=5.0` |
| Épocas | 100 |
| Batch size | 64 |
| Semente | 42 |
| Device | CUDA se disponível, senão CPU |

A loss de treino espelha a métrica de avaliação: otimiza o erro absoluto ponderado que o requisito RNF-01 cobra. O clipping limita o impacto dos outliers de venda no gradiente.

Ao fim de cada época o WMAE de teste é calculado; os pesos da melhor época são guardados e restaurados no final.

## Métrica

O **WMAE** (`evaluation/metrics.py`) é a métrica principal, herdada da competição original do dataset e formalizada no GR4ML como RNF-01 e RNF-04:

```
WMAE = Σ(wᵢ · |yᵢ − ŷᵢ|) / Σwᵢ,  onde wᵢ = 5 em semana de feriado e 1 nas demais
```

## Testes

Suíte em `unittest`:

```bash
python -m unittest discover -s tests
```

| Arquivo | Cobre |
| --- | --- |
| `test_data_loader.py` | leitura do CSV e validação de base vazia |
| `test_preprocessing.py` | limpeza, atributos temporais, split e normalização |
| `test_datasets.py` | shapes e dtypes dos tensores, dataset e batches |
| `test_model.py` | shape e finitude da saída das redes |
| `test_save_load.py` | formato do checkpoint e round-trip dos pesos |
| `test_system_rules.py` | split cronológico e ausência de vazamento na normalização |

`tests/helpers.py` gera um DataFrame sintético com o mesmo schema do dataset real.

## Limitações conhecidas

TODO: Gabriel Leite

## Equipe

Luis Eduardo Moreira Las Casas · Thiago Augusto Souza do Nascimento · Iuan Ferraz Souza · Natália Michelini Monteiro · Gabriela Ladeira Nadaes · Gabriel Dantas Leite · Gabriel Lucena

---

Programa de Pós-Graduação em Deep Learning — Centro de Informática (CIn), Universidade Federal de Pernambuco (UFPE)
