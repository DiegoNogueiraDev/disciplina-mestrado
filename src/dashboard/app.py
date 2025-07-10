import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List
import glob
import os

logger = logging.getLogger(__name__)

def create_dashboard_app(data_path: str = "data/output") -> dash.Dash:
    """
    Criar aplicação Dash para dashboard de sentimentos
    
    Args:
        data_path: Caminho para os dados de saída
        
    Returns:
        Aplicação Dash configurada
    """
    app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
    
    # Layout do dashboard
    app.layout = html.Div([
        html.H1("Dashboard de Análise de Sentimento - X/Twitter vs Reddit", 
                style={'textAlign': 'center', 'marginBottom': 30}),
        
        # Controles
        html.Div([
            html.Div([
                html.Label("Selecionar Dataset:"),
                dcc.Dropdown(
                    id='dataset-dropdown',
                    options=[],
                    value=None,
                    clearable=False
                )
            ], className='four columns'),
            
            html.Div([
                html.Label("Filtrar por Plataforma:"),
                dcc.Dropdown(
                    id='platform-filter',
                    options=[
                        {'label': 'Todas', 'value': 'all'},
                        {'label': 'Twitter', 'value': 'twitter'},
                        {'label': 'Reddit', 'value': 'reddit'}
                    ],
                    value='all',
                    clearable=False
                )
            ], className='four columns'),
            
            html.Div([
                html.Label("Confiança Mínima:"),
                dcc.Slider(
                    id='confidence-slider',
                    min=0,
                    max=1,
                    step=0.1,
                    value=0.5,
                    marks={i/10: str(i/10) for i in range(0, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], className='four columns'),
        ], className='row', style={'marginBottom': 30}),
        
        # Métricas principais
        html.Div([
            html.Div([
                html.H4("Total de Posts"),
                html.H2(id='total-posts', children="0")
            ], className='three columns', style={'textAlign': 'center', 'backgroundColor': '#f9f9f9', 'padding': 20}),
            
            html.Div([
                html.H4("Sentimento Positivo"),
                html.H2(id='positive-percentage', children="0%")
            ], className='three columns', style={'textAlign': 'center', 'backgroundColor': '#d4edda', 'padding': 20}),
            
            html.Div([
                html.H4("Sentimento Neutro"),
                html.H2(id='neutral-percentage', children="0%")
            ], className='three columns', style={'textAlign': 'center', 'backgroundColor': '#fff3cd', 'padding': 20}),
            
            html.Div([
                html.H4("Sentimento Negativo"),
                html.H2(id='negative-percentage', children="0%")
            ], className='three columns', style={'textAlign': 'center', 'backgroundColor': '#f8d7da', 'padding': 20}),
        ], className='row', style={'marginBottom': 30}),
        
        # Gráficos
        html.Div([
            html.Div([
                dcc.Graph(id='sentiment-timeline')
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(id='platform-comparison')
            ], className='six columns'),
        ], className='row', style={'marginBottom': 30}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='confidence-distribution')
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(id='sentiment-distribution')
            ], className='six columns'),
        ], className='row', style={'marginBottom': 30}),
        
        # Tabela de amostras
        html.Div([
            html.H3("Amostras de Posts por Sentimento"),
            html.Div(id='sample-posts-table')
        ], style={'marginBottom': 30}),
        
        # Intervalo de atualização
        dcc.Interval(
            id='interval-component',
            interval=60*1000,  # Atualizar a cada 60 segundos
            n_intervals=0
        )
    ])
    
    @app.callback(
        Output('dataset-dropdown', 'options'),
        Output('dataset-dropdown', 'value'),
        Input('interval-component', 'n_intervals')
    )
    def update_dataset_options(n):
        """Atualizar opções de dataset"""
        try:
            # Buscar arquivos CSV na pasta de saída
            csv_files = glob.glob(os.path.join(data_path, "*.csv"))
            
            if not csv_files:
                return [], None
            
            options = []
            for file_path in csv_files:
                filename = os.path.basename(file_path)
                options.append({'label': filename, 'value': file_path})
            
            # Ordenar por data de modificação (mais recente primeiro)
            options.sort(key=lambda x: os.path.getmtime(x['value']), reverse=True)
            
            return options, options[0]['value'] if options else None
            
        except Exception as e:
            logger.error(f"Erro ao buscar datasets: {e}")
            return [], None
    
    @app.callback(
        [Output('total-posts', 'children'),
         Output('positive-percentage', 'children'),
         Output('neutral-percentage', 'children'),
         Output('negative-percentage', 'children'),
         Output('sentiment-timeline', 'figure'),
         Output('platform-comparison', 'figure'),
         Output('confidence-distribution', 'figure'),
         Output('sentiment-distribution', 'figure'),
         Output('sample-posts-table', 'children')],
        [Input('dataset-dropdown', 'value'),
         Input('platform-filter', 'value'),
         Input('confidence-slider', 'value')]
    )
    def update_dashboard(dataset_path, platform_filter, confidence_threshold):
        """Atualizar todos os componentes do dashboard"""
        if not dataset_path:
            empty_fig = go.Figure()
            return "0", "0%", "0%", "0%", empty_fig, empty_fig, empty_fig, empty_fig, "Nenhum dado disponível"
        
        try:
            # Carregar dados
            df = pd.read_csv(dataset_path)
            
            # Filtrar por plataforma
            if platform_filter != 'all':
                df = df[df['platform'] == platform_filter]
            
            # Filtrar por confiança
            if 'confidence' in df.columns:
                df = df[df['confidence'] >= confidence_threshold]
            
            if df.empty:
                empty_fig = go.Figure()
                return "0", "0%", "0%", "0%", empty_fig, empty_fig, empty_fig, empty_fig, "Nenhum dado após filtros"
            
            # Calcular métricas
            total_posts = len(df)
            sentiment_counts = df['predicted_sentiment'].value_counts()
            
            pos_pct = f"{sentiment_counts.get('positive', 0) / total_posts * 100:.1f}%"
            neu_pct = f"{sentiment_counts.get('neutral', 0) / total_posts * 100:.1f}%"
            neg_pct = f"{sentiment_counts.get('negative', 0) / total_posts * 100:.1f}%"
            
            # Converter timestamp para datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Gráfico de linha temporal
            timeline_fig = create_sentiment_timeline(df)
            
            # Gráfico de comparação por plataforma
            platform_fig = create_platform_comparison(df)
            
            # Gráfico de distribuição de confiança
            confidence_fig = create_confidence_distribution(df)
            
            # Gráfico de distribuição de sentimentos
            sentiment_fig = create_sentiment_distribution(df)
            
            # Tabela de amostras
            sample_table = create_sample_table(df)
            
            return (str(total_posts), pos_pct, neu_pct, neg_pct, 
                   timeline_fig, platform_fig, confidence_fig, sentiment_fig, sample_table)
            
        except Exception as e:
            logger.error(f"Erro ao processar dados: {e}")
            empty_fig = go.Figure()
            return "Erro", "Erro", "Erro", "Erro", empty_fig, empty_fig, empty_fig, empty_fig, f"Erro: {str(e)}"
    
    return app

def create_sentiment_timeline(df: pd.DataFrame) -> go.Figure:
    """Criar gráfico de linha temporal de sentimentos"""
    # Agrupar por hora e sentimento
    df_hourly = df.groupby([df['timestamp'].dt.floor('H'), 'predicted_sentiment']).size().reset_index(name='count')
    
    fig = px.line(df_hourly, x='timestamp', y='count', color='predicted_sentiment',
                  title='Evolução Temporal dos Sentimentos',
                  labels={'count': 'Número de Posts', 'timestamp': 'Tempo'},
                  color_discrete_map={'positive': '#28a745', 'neutral': '#ffc107', 'negative': '#dc3545'})
    
    fig.update_layout(hovermode='x unified')
    return fig

def create_platform_comparison(df: pd.DataFrame) -> go.Figure:
    """Criar gráfico de comparação por plataforma"""
    if 'platform' not in df.columns:
        return go.Figure()
    
    # Calcular percentuais por plataforma
    platform_sentiment = df.groupby(['platform', 'predicted_sentiment']).size().unstack(fill_value=0)
    platform_sentiment_pct = platform_sentiment.div(platform_sentiment.sum(axis=1), axis=0) * 100
    
    fig = go.Figure()
    
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in platform_sentiment_pct.columns:
            fig.add_trace(go.Bar(
                name=sentiment.capitalize(),
                x=platform_sentiment_pct.index,
                y=platform_sentiment_pct[sentiment],
                marker_color={'positive': '#28a745', 'neutral': '#ffc107', 'negative': '#dc3545'}[sentiment]
            ))
    
    fig.update_layout(
        title='Distribuição de Sentimentos por Plataforma',
        xaxis_title='Plataforma',
        yaxis_title='Percentual (%)',
        barmode='stack'
    )
    
    return fig

def create_confidence_distribution(df: pd.DataFrame) -> go.Figure:
    """Criar gráfico de distribuição de confiança"""
    if 'confidence' not in df.columns:
        return go.Figure()
    
    fig = px.histogram(df, x='confidence', nbins=20, 
                      title='Distribuição de Confiança das Predições',
                      labels={'confidence': 'Confiança', 'count': 'Frequência'})
    
    fig.add_vline(x=df['confidence'].mean(), line_dash="dash", 
                  annotation_text=f"Média: {df['confidence'].mean():.2f}")
    
    return fig

def create_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """Criar gráfico de distribuição de sentimentos"""
    sentiment_counts = df['predicted_sentiment'].value_counts()
    
    fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                 title='Distribuição Geral de Sentimentos',
                 color_discrete_map={'positive': '#28a745', 'neutral': '#ffc107', 'negative': '#dc3545'})
    
    return fig

def create_sample_table(df: pd.DataFrame, n_samples: int = 5) -> html.Table:
    """Criar tabela de amostras"""
    samples = []
    
    for sentiment in ['positive', 'neutral', 'negative']:
        sentiment_df = df[df['predicted_sentiment'] == sentiment]
        if not sentiment_df.empty:
            sample = sentiment_df.sample(min(n_samples, len(sentiment_df)))
            samples.append(sample)
    
    if not samples:
        return html.Div("Nenhuma amostra disponível")
    
    sample_df = pd.concat(samples)
    
    # Criar tabela HTML
    table_header = [
        html.Thead([
            html.Tr([
                html.Th("Plataforma"),
                html.Th("Sentimento"),
                html.Th("Confiança"),
                html.Th("Texto (primeiras 100 chars)"),
                html.Th("Timestamp")
            ])
        ])
    ]
    
    table_body = [
        html.Tbody([
            html.Tr([
                html.Td(row['platform'] if 'platform' in row else 'N/A'),
                html.Td(row['predicted_sentiment'], 
                       style={'color': {'positive': '#28a745', 'neutral': '#ffc107', 'negative': '#dc3545'}[row['predicted_sentiment']]}),
                html.Td(f"{row['confidence']:.2f}" if 'confidence' in row else 'N/A'),
                html.Td(row['lemmatized'][:100] + "..." if len(row['lemmatized']) > 100 else row['lemmatized']),
                html.Td(row['timestamp'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['timestamp']) else 'N/A')
            ]) for _, row in sample_df.iterrows()
        ])
    ]
    
    return html.Table(table_header + table_body, className='table')

if __name__ == '__main__':
    app = create_dashboard_app()
    app.run_server(debug=True, host='0.0.0.0', port=8050)