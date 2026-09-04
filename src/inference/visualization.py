import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

def plot_financial_projection(df_historical: pd.DataFrame, df_future: pd.DataFrame, store_id: int, dept_id: int):
    """Gera um gráfico financeiro comparando vendas históricas e projeções futuras."""
        
    historical_filtered = df_historical[(df_historical['Store'] == store_id) & (df_historical['Dept'] == dept_id)].copy()
    historical_filtered['Date'] = pd.to_datetime(historical_filtered['Date'])
    historical_filtered.set_index('Date', inplace=True)
    
    # Agrupa os dados semanais por mês (final do mês - 'ME') para visualização executiva
    historical_monthly = historical_filtered['Weekly_Sales'].resample('ME').sum()
    
    # Preparação dos dados futuros
    future_filtered = df_future.copy()
    future_filtered['Date'] = pd.to_datetime(future_filtered['Date'])
    future_filtered.set_index('Date', inplace=True)
    
    future_monthly = future_filtered['Projected_Sales'].resample('ME').sum()
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Histórico (Azul escuro sólido)
    ax.plot(historical_monthly.index, historical_monthly.values, 
            label='Vendas Realizadas (Histórico)', color='#1b4f72', linewidth=2.5)
    
    # Futuro (Laranja tracejado)
    ax.plot(future_monthly.index, future_monthly.values, 
            label='Projeção IA (Futuro)', color='#e67e22', linewidth=2.5, linestyle='--')
    
    # Conecta visualmente o último ponto histórico ao primeiro ponto futuro
    if not historical_monthly.empty and not future_monthly.empty:
        ax.plot([historical_monthly.index[-1], future_monthly.index[0]], 
                [historical_monthly.values[-1], future_monthly.values[0]], 
                color='#e67e22', linewidth=2.5, linestyle='--')
    
    ax.set_title(f'Projeção Financeira Mensal - Loja {store_id} / Dept {dept_id}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Período', fontsize=12, labelpad=10)
    ax.set_ylabel('Receita Total ($)', fontsize=12, labelpad=10)
    
    formatter = ticker.FuncFormatter(lambda x, pos: f'${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    
    sns.despine(top=True, right=True)
    
    ax.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    
    plt.show()