#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install dash pandas geopandas plotly3ywr.t5')


# In[3]:


import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import chardet
import requests
import chardet
import plotly.graph_objects as go
import seaborn as sns
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import geopandas as gpd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import geopandas as gpd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px


# In[5]:


gdf_cafe = gpd.read_file("C:/analysis/깃헙/FILE1/data/스타벅스 빽다방 개수, 밀도.shp")
gdf_price = gpd.read_file("C:/analysis/깃헙/FILE1/data/아파트,토지 가격.shp")
gdf_sub = gpd.read_file("C:/analysis/깃헙/FILE1/data/교통 인프라.shp")
gdf_work = gpd.read_file("C:/analysis/깃헙/FILE1/data/타지역 통근.shp")
gdf_hos = gpd.read_file("C:/analysis/기말발표/성형외과 내과 밀도/성형외과 내과 밀도/성형외과.shp")


# In[6]:


import geopandas as gpd
from fiona.crs import from_epsg

# gdf_cafe를 적절한 좌표계로 변환
gdf_cafe = gdf_cafe.to_crs(epsg=4326)
gdf_sub = gdf_sub.to_crs(epsg=4326)
gdf_work = gdf_work.to_crs(epsg=4326)
gdf_hos = gdf_hos.to_crs(epsg=4326)


# In[7]:


app = dash.Dash(__name__)

# Define layout for the tabs
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='단계구분도', children=[
            html.H1("단계구분도"),
            dcc.Dropdown(
                id='choropleth-dropdown',
                options=[
                    {'label': 'Starbucks', 'value': 'Starbucks'},
                    {'label': 'Subway', 'value': 'Subway'},
                    {'label': 'Work', 'value': 'Work'},
                    {'label': 'Hospital', 'value' : 'Hospital'}
                ],
                value='Starbucks',
                style={'width': '200px'}
            ),
            dcc.Graph(id='choropleth-map', style={'height': '800px'})  # 수정된 부분: 'choropleth-chart' -> 'choropleth-map'
        ]),
        
                
       dcc.Tab(label='상하위 3개구 파이차트', children=[
            html.H1("상하위 3개구 파이차트"),
            dcc.Dropdown(
                id='pie-dropdown',
                options=[
                    {'label': 'Starbucks', 'value': 'Starbucks'},
                    {'label': 'Subway', 'value': 'Subway'},
                    {'label': 'Work', 'value': 'Work'},
                    {'label': 'Hospital', 'value' : 'Hospital'}
                ],
                value='Starbucks',
                style={'width': '200px'}
            ),
            dcc.Graph(id='pie-chart')
        ]),
        
        dcc.Tab(label='상관관계 분석', children=[
            html.H1("상관관계 분석"),
            dcc.Dropdown(
                id='correlation-dropdown',
                options=[
                    {'label': 'Starbucks', 'value': 'Starbucks'},
                    {'label': 'Subway', 'value': 'Subway'},
                    {'label': 'Work', 'value': 'Work'},
                    {'label': 'Hospital', 'value' : 'Hospital'}
                ],
                value='Starbucks',
                style={'width': '200px'}
            ),
            dcc.Graph(id='correlation-graph')
        ]),
        
    ])
])
# Callbacks
@app.callback(
    Output('choropleth-map', 'figure'),  # 수정된 부분: 'choropleth-chart' -> 'choropleth-map'
    [Input('choropleth-dropdown', 'value')]
)
def update_choropleth_map(selected_item):
    if selected_item == 'Starbucks':
        geo_df = gdf_cafe
        geo_column = 'STAR densi'
    elif selected_item == 'Subway':
        geo_df = gdf_sub
        geo_column = 'sub_densit'
    elif selected_item == 'Work':
        geo_df = gdf_work
        geo_column = 'density'
    elif selected_item == 'Hospital':
        geo_df = gdf_hos
        geo_column = 'density'
    
    fig = px.choropleth_mapbox(
        geo_df,
        geojson=geo_df.geometry,
        locations=geo_df.index.astype(str),
        color=geo_df[geo_column],
        color_continuous_scale='YlGnBu',
        range_color=(geo_df[geo_column].min(), geo_df[geo_column].max()),
        mapbox_style="carto-positron",
        center={"lat": 37.5665, "lon": 126.9780},
        zoom=10,
        opacity=0.5,
        labels={geo_column: f'{selected_item} 밀도'}
    )
    fig.update_layout(title=f'{selected_item} 밀도 단계 구분도')
    return fig

@app.callback(
    Output('correlation-graph', 'figure'),
    [Input('correlation-dropdown', 'value')]
)
def update_correlation_graph(selected_item):
    if selected_item == 'Starbucks':
        x_data = gdf_cafe['STAR densi']
    elif selected_item == 'Subway':
        x_data = gdf_sub['sub_densit']
    elif selected_item == 'Work':
        x_data = gdf_work['density']
    elif selected_item == 'Hospital':
        x_data = gdf_hos['density']
    correlation_coefficient = np.corrcoef(gdf_price['land_price'], x_data)[0, 1]

    fig = px.scatter(x=gdf_price['land_price'], y=x_data, labels={'x': '공시지가', 'y': f'{selected_item} 밀도'}, trendline="ols")
   
    fig.update_layout(
        title=f'{selected_item}와 공시지가의 상관관계',
        annotations=[dict(
            x=0.5,
            y=0.9,
            showarrow=False,
            text=f'상관계수: {correlation_coefficient:.2f}',
            xref='paper',
            yref='paper'
        )]
    )
    
    return fig


@app.callback(
    Output('pie-chart', 'figure'),
    [Input('pie-dropdown', 'value')]
)
def update_pie_chart(selected_item):
    global gdf_sub, gdf_work, gdf_hos  

    if selected_item == 'Starbucks':
        sorted_gdf = gdf_cafe.groupby('SIG_KOR_NM')['STAR densi'].sum().reset_index()
    elif selected_item == 'Subway':
        sorted_gdf = gdf_sub.groupby('SIG_KOR_NM')['sub_densit'].sum().reset_index()
    elif selected_item == 'Work':
        sorted_gdf = gdf_work.groupby('SIG_KOR_NM')['density'].sum().reset_index()
    elif selected_item == 'Hospital':
        sorted_gdf = gdf_hos.groupby('SIG_KOR_NM')['density'].sum().reset_index()
    else:
        return {'data': [], 'layout': {}}

    selected_regions = ['강남구', '서초구', '중구', '도봉구', '노원구', '강북구']
    sorted_gdf = sorted_gdf[sorted_gdf['SIG_KOR_NM'].isin(selected_regions)]

    trace = go.Pie(
        labels=sorted_gdf['SIG_KOR_NM'],
        values=sorted_gdf[sorted_gdf.columns[1]],  # 각각의 데이터프레임에 따라 열 선택
        pull=[0.2, 0, 0, 0, 0, 0],  # 필요에 따라 조절
        hole=0.3  # 필요에 따라 조절
    )
    layout = go.Layout(title=f'{selected_item} 파이 차트')
    return {'data': [trace], 'layout': layout}
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

