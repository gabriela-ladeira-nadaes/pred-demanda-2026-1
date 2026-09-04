import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

def plot_financial_projection(df_historical: pd.DataFrame, df_future: pd.DataFrame, store_id: int= None, dept_id: int= None):
    """Gera um gráfico financeiro comparando vendas históricas e projeções futuras."""
        
    # Define os títulos e filtros baseado nos parâmetros
    if store_id is not None and dept_id is not None:
        title_suffix = f"Loja {store_id} / Dept {dept_id}"
        hist_filtered = df_historical[(df_historical['Store'] == store_id) & (df_historical['Dept'] == dept_id)].copy()
    else:
        title_suffix = "Rede Walmart Completa"
        hist_filtered = df_historical.copy()

    hist_filtered['Date'] = pd.to_datetime(hist_filtered['Date'])
    hist_filtered.set_index('Date', inplace=True)
    historical_monthly = hist_filtered['Weekly_Sales'].resample('ME').sum()
    
    future_filtered = df_future.copy()
    future_filtered['Date'] = pd.to_datetime(future_filtered['Date'])
    future_filtered.set_index('Date', inplace=True)
    future_monthly = future_filtered['Projected_Sales'].resample('ME').sum()
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(historical_monthly.index, historical_monthly.values, 
            label='Vendas Realizadas (Histórico)', color='#1b4f72', linewidth=2.5)
    
    ax.plot(future_monthly.index, future_monthly.values, 
            label='Projeção IA (Futuro)', color='#e67e22', linewidth=2.5, linestyle='--')
    
    if not historical_monthly.empty and not future_monthly.empty:
        ax.plot([historical_monthly.index[-1], future_monthly.index[0]], 
                [historical_monthly.values[-1], future_monthly.values[0]], 
                color='#e67e22', linewidth=2.5, linestyle='--')
    
    ax.set_title(f'Projeção Financeira Mensal - {title_suffix}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Período', fontsize=12, labelpad=10)
    ax.set_ylabel('Receita Total ($)', fontsize=12, labelpad=10)
    
    formatter = ticker.FuncFormatter(lambda x, pos: f'${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    sns.despine(top=True, right=True)
    ax.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_financial_acumulate_projection(df_historical: pd.DataFrame, df_future: pd.DataFrame, store_id: int = None, dept_id: int = None):
    """Gera um gráfico combinado: barras (faturamento mensal) e linhas (receita acumulada)."""
    
    historical_monthly, future_monthly, title_suffix = _prepare_data(df_historical, df_future, store_id, dept_id)
    
    # Cálculo do acumulado
    historical_cumulative = historical_monthly.cumsum()
    
    if not historical_monthly.empty:
        last_hist_value = historical_cumulative.iloc[-1]
        future_cumulative = future_monthly.cumsum() + last_hist_value
    else:
        future_cumulative = future_monthly.cumsum()

    # Eixos Duplos
    sns.set_theme(style="white")
    fig, ax1 = plt.subplots(figsize=(15, 7))
    ax2 = ax1.twinx() 
    
    bar_width = 20
    
    # Barras (Mensal) - Eixo 1
    ax1.bar(historical_monthly.index, historical_monthly.values, 
            width=bar_width, label='Mensal (Realizado)', color='#1b4f72', alpha=0.4)
    ax1.bar(future_monthly.index, future_monthly.values, 
            width=bar_width, label='Mensal (Projetado)', color='#e67e22', alpha=0.4)

    # Linhas (Acumulado) - Eixo 2
    ax2.plot(historical_cumulative.index, historical_cumulative.values, 
             label='Acumulado (Realizado)', color='#1b4f72', linewidth=3, marker='o')
    ax2.plot(future_cumulative.index, future_cumulative.values, 
             label='Acumulado (Projetado)', color='#e67e22', linewidth=3, linestyle='--', marker='o')
    
    # Conexão visual do acumulado
    if not historical_monthly.empty and not future_monthly.empty:
        ax2.plot([historical_cumulative.index[-1], future_cumulative.index[0]], 
                 [historical_cumulative.values[-1], future_cumulative.values[0]], 
                 color='#e67e22', linewidth=3, linestyle='--')

    # Formatação
    ax1.set_title(f'Projeção Financeira Acumulada - {title_suffix}', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Período', fontsize=12, labelpad=10)
    ax1.set_ylabel('Faturamento do Mês ($)', fontsize=12, color='#555555')
    ax2.set_ylabel('Receita Acumulada ($)', fontsize=12, fontweight='bold')
    
    formatter = ticker.FuncFormatter(lambda x, pos: f'${x:,.0f}')
    ax1.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)
    
    # Estica as barras para baixo
    max_monthly = max(historical_monthly.max(), future_monthly.max()) if not future_monthly.empty else historical_monthly.max()
    ax1.set_ylim(0, max_monthly * 3.5) 
    
    # Legenda combinada
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11, framealpha=0.95)
    
    ax1.grid(False) 
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    sns.despine(top=True, right=False)
    
    plt.tight_layout()
    plt.show()

def _prepare_data(df_historical: pd.DataFrame, df_future: pd.DataFrame, store_id: int = None, dept_id: int = None):
    """Função auxiliar para filtrar e agrupar os dados mensalmente."""
    if store_id is not None and dept_id is not None:
        title_suffix = f"Loja {store_id} / Dept {dept_id}"
        hist_filtered = df_historical[(df_historical['Store'] == store_id) & (df_historical['Dept'] == dept_id)].copy()
    else:
        title_suffix = "Rede Walmart Completa"
        hist_filtered = df_historical.copy()

    hist_filtered['Date'] = pd.to_datetime(hist_filtered['Date'])
    hist_filtered.set_index('Date', inplace=True)
    historical_monthly = hist_filtered['Weekly_Sales'].resample('ME').sum()
    
    future_filtered = df_future.copy()
    future_filtered['Date'] = pd.to_datetime(future_filtered['Date'])
    future_filtered.set_index('Date', inplace=True)
    future_monthly = future_filtered['Projected_Sales'].resample('ME').sum()
    
    return historical_monthly, future_monthly, title_suffix