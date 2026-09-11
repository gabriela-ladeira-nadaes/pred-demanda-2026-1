# Predição de Demanda — Vendas Semanais do Walmart

Projeto da disciplina **Engenharia de Software para IA e Frameworks Profundos** do programa de pós-graduação em Deep Learning do Centro de Informática (CIn) da Universidade Federal de Pernambuco (UFPE).

## Problema e objetivo

As variações semanais das vendas dificultam o planejamento de abastecimento das lojas. O sistema aprende os padrões históricos de venda por loja e departamento — junto com feriados, promoções, características da loja e indicadores econômicos — para projetar as vendas das próximas semanas e apoiar a priorização do abastecimento.

O levantamento de requisitos foi feito com o framework **GR4ML** (documento `predicao_demandas_GR4ML.pdf`, na raiz do repositório), que define o WMAE como métrica principal e origina os requisitos funcionais, não funcionais e de dados.

A apresentação final está em `predicao_demandas_apresentacao.pdf`, também na raiz.

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

**`LSTMModel`** — LSTM de 2 camadas com 64 unidades ocultas, seguida de uma camada linear. Como cada exemplo é uma linha independente, a entrada é tratada como uma sequência de comprimento 1.

**`LinearRegression`** — uma única camada `Linear(input, 1)`, usada como baseline.

A entrada tem 200 dimensões: as 17 colunas de features viram 200 após o one-hot de loja, departamento, tipo, mês e semana do ano.

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

## Configuração experimental

Split cronológico com 294.132 registros de treino (fev/2010 – dez/2011) e 127.438 de teste (jan/2012 – out/2012). Os três modelos são treinados com os mesmos hiperparâmetros e comparados pelo WMAE de teste.

**O que buscávamos descobrir:** se alinhar a loss de treino à métrica de negócio (WMAE) e limitar a norma do gradiente reduziria a oscilação do erro entre épocas, sem comprometer a convergência.

| Dimensão | Configurações testadas |
| --- | --- |
| Função de perda | MSE × L1 ponderada, alinhada ao WMAE (adotada) |
| Gradient clipping | sem clipping × `max_norm = 1,0` × `max_norm = 5,0` (adotado) |
| Modelo | Regressão Linear (baseline) × LSTM × MLP |

## Resultados

Valores em dólares, após desfazer a normalização do alvo, na melhor época de cada modelo. Execução reportada na apresentação.

| Modelo | Conjunto | RMSE | MAE | WMAE | R² |
| --- | --- | ---: | ---: | ---: | ---: |
| Regressão Linear | treino | 14.521,13 | 7.411,99 | 7.680,21 | 0,6 |
| | teste | 13.049,88 | 7.126,73 | 7.193,12 | 0,652 |
| **MLP** | treino | 6.187,18 | 2.895,19 | 3.050,73 | 0,9274 |
| | **teste** | **5.183,67** | **2.534,74** | **2.604,71** | **0,9451** |
| LSTM | treino | 4.009,26 | 1.647,18 | 1.715,51 | 0,9695 |
| | teste | 5.846,49 | 3.199,35 | 3.351,25 | 0,9301 |

- **O MLP é o modelo escolhido** e o artefato salvo em `models/`: menor WMAE de teste, 63,8% abaixo da regressão linear.
- **O LSTM sobreajusta.** Tem o menor erro de treino (WMAE 1.715,51), mas piora no teste (3.351,25) — a capacidade extra não generaliza para 2012.
- **A regressão linear fica em R² 0,65 no teste**, o que confirma que a relação entre os atributos e as vendas não é linear.
- O MLP acompanha bem o nível agregado das vendas, mas responde mal a picos pontuais de uma única semana.

O `src/predict.py` usa o modelo escolhido para projetar 52 semanas da rede completa e gera os gráficos de projeção mensal e acumulada.

## Atendimento aos requisitos

| Requisito | Situação | Evidência |
| --- | --- | --- |
| RF01 — previsão semanal por loja e departamento | Atendido | `generate_forecast` projeta N semanas para todas as combinações |
| RF02 — consulta por loja, departamento ou rede | Atendido | parâmetros `store_id` e `dept_id` de `generate_forecast` |
| RF03 — ordenar por maiores e menores vendas previstas | Não implementado | próximo passo |
| RF04 — registrar o realizado e comparar com o previsto | Não implementado | próximo passo |
| RF05 — comparar com a semana anterior e o histórico | Parcial | gráficos de histórico × projeção em `visualization.py`, sem comparação tabular |
| RNF01 — reduzir o WMAE em ≥ 15% em relação ao baseline | Atendido | −63,8% em relação à regressão linear (7.193,12 → 2.604,71) |
| RNF02 — reprodutibilidade, diferença ≤ 5% entre execuções | Parcial | semente fixa em `config.py`; comparação entre execuções não automatizada |
| RNF03 — 100% de cobertura, sem previsões inválidas | Parcial | uma linha por combinação × semana, mas 0,10% das projeções saem negativas |
| RNF04 — componentes testáveis isoladamente | Parcial | carregamento, preparação temporal e persistência têm testes; WMAE e inferência não |
| RNF05 — ponderar feriados | Atendido | WMAE com peso 5 na avaliação e na loss de treino |

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

- **Dificuldade em janelas.** MLP que responde bem globalmente, mas não é bom para entender impactos temporais. Ex. Pico de venda em uma semana 

## Próximos passos

- Testar modelos baseados em árvore, como XGBoost e Random Forest.
- Automatizar o retreino junto ao fechamento mensal.

## Equipe

Luis Eduardo Moreira Las Casas · Thiago Augusto Souza do Nascimento · Iuan Ferraz Souza · Natália Michelini Monteiro · Gabriela Ladeira Nadaes · Gabriel Dantas Leite · Gabriel Lucena

---

Programa de Pós-Graduação em Deep Learning — Centro de Informática (CIn), Universidade Federal de Pernambuco (UFPE)
