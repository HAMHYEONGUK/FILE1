#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install dash pandas geopandas plotly3ywr.t5')


# In[9]:


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


# In[10]:


gdf_cafe = gpd.read_file("data/스타벅스 빽다방 개수, 밀도.shp")
gdf_price = gpd.read_file("data/아파트,토지 가격.shp")
gdf_sub = gpd.read_file("data/교통 인프라.shp")
gdf_work = gpd.read_file("data/타지역 통근.shp")
gdf_hos = gpd.read_file("data/성형외과.shp")


# In[11]:


import geopandas as gpd
from fiona.crs import from_epsg

# gdf_cafe를 적절한 좌표계로 변환
gdf_cafe = gdf_cafe.to_crs(epsg=4326)
gdf_sub = gdf_sub.to_crs(epsg=4326)
gdf_work = gdf_work.to_crs(epsg=4326)
gdf_hos = gdf_hos.to_crs(epsg=4326)


# In[12]:


app = dash.Dash(__name__)

# Define layout for the tabs
app.layout = html.Div([
    html.H1("서울시 공시지가로 보는 공간적 불평등"),
    dcc.Tabs([
        dcc.Tab(label='문화적 측면', children=[
            html.P("이는 문화적 측면에 공간적 불평등을 설명하는 탭입니다."),
             dcc.Dropdown(
                id='choropleth-dropdown-starbucks',
                options=[
                    {'label': 'Starbucks', 'value': 'Starbucks'},
                ],
                value='Starbucks',
                disabled=True,  # 드롭다운을 사용 불가능하게 만듦
                style={'width': '200px'}
            ),
            dcc.Graph(id='choropleth-map', style={'height': '680px'}),  # 수정된 부분: 'choropleth-chart' -> 'choropleth-map'
            html.Div([
                dcc.Graph(id='bar-chart', style={'width': '100%', 'display': 'inline-block'})
            ]),
           
            # 왼쪽 오른쪽으로 각각 50% 차지하는 두 개의 상관관계 분석 그래프
            html.Div([
                dcc.Graph(id='correlation-graph1', style={'width': '50%', 'display': 'inline-block', 'height': '500px'}),
                dcc.Graph(id='correlation-graph2', style={'width': '50%', 'display': 'inline-block', 'height': '500px'})
            ]),
            html.P("공시지가와 스타벅스 분포 밀도를 상관관계 분석한 결과 0.69로 높은 상관관계가 나타나는 것을 볼 수 있음."
                   "카페 입지 조건이 유동인구가 많고 상권이 발달한 곳 즉, 공시지가가 높은 곳에 입지하는 것은 사실이지만 모든 카페가 그 경우에 해당하는 것은 아님."
                   "위의 자료를 통해 빽다방의 경우에는 공시지가와의 상관관계 분석에서 상관계수 -0.09로 상관관계를 띄지 않는데 이는 스타벅스 밀도와 공시지가와의 상관관계 분석이 문화적인 측면의 공간적 불평등을 초래한다는 결과를 뒷받침 할 수 있다고 판단하였음."),
        ]),

#스타벅스 레이아웃
                
       dcc.Tab(label='교통적 측면', children=[
                html.H1(),
                html.P("이는 교통적 측면에 공간적 불평등을 설명하는 탭입니다."),
                 dcc.Dropdown(
                    id='choropleth-dropdown-sub',
                    options=[
                    {'label': 'Subway', 'value': 'Subway'},
                    ],
                    value='Subway',
                    disabled=True,  # 드롭다운을 사용 불가능하게 만듦
                    style={'width': '200px'}
                ),
                dcc.Graph(id='choropleth-map_sub', style={'height': '680px'}),  # 수정된 부분: 'choropleth-chart' -> 'choropleth-map'
                html.Div([
                    dcc.Graph(id='bar-chart_sub', style={'width': '100%', 'display': 'inline-block'})
                ]),
            # 왼쪽 오른쪽으로 각각 50% 차지하는 두 개의 상관관계 분석 그래프
                html.Div([
                    dcc.Graph(id='correlation-graph_sub', style={'width': '50%', 'display': 'inline-block', 'height': '500px'}),
                    dcc.Graph(id='correlation-graph_sub2', style={'width': '50%', 'display': 'inline-block', 'height': '500px'})
                ]),
             html.P("공시지가와 지하철 분포 밀도와의 상관관계 분석 결과는 0.70으로 높은 상관관계가 나타남."
                    "지하철 역 입지 조건이 유동인구가 많고 상권이 발달한 곳 즉, 공시지가가 높은 곳에 입지하는 것은 사실이지만 모든 교통체계가 그 경우에 해당하는 것은 아님."
                    "위의 자료를 통해 버스정류장의 경우에는 공시지가와의 상관관계 분석에서 상관계수 -0.2로 상관관계를 띄지 않는데 이는 지하철역 밀도와 공시지가와의 상관관계 분석이 교통적인 측면의 공간적 불평등을 초래한다는 결과를 뒷받침 할 수 있다고 판단하였음."),
            
            ]),
#지하철 레이아웃     
        dcc.Tab(label='주거적 측면', children=[
                html.H1(),
                html.P("이는 주거적 측면에 공간적 불평등을 설명하는 탭입니다."),
                 dcc.Dropdown(
                    id='choropleth-dropdown-work',
                    options=[
                    {'label': 'Work', 'value': 'Work'},
                    ],
                    value='Work',
                    disabled=True,  # 드롭다운을 사용 불가능하게 만듦
                    style={'width': '200px'}
                ),
                dcc.Graph(id='choropleth-map_work', style={'height': '680px'}),  # 수정된 부분: 'choropleth-chart' -> 'choropleth-map'
                html.Div([
                    dcc.Graph(id='bar-chart_work', style={'width': '100%', 'display': 'inline-block'})
                ]),
            # 왼쪽 오른쪽으로 각각 50% 차지하는 두 개의 상관관계 분석 그래프
                html.Div([
                    dcc.Graph(id='correlation-graph_work', style={'width': '100%', 'display': 'inline-block', 'height': '500px'})
                ]),
            html.P("여기서 말하는 통근의 개념은, 예를 들어 강남구에서 서초구로 구에서 구로 통근하는 사람들의 밀도를 정리한 데이터임."
                    "상위 시군구에 위와는 다소 다른 동작구 양천구 광진구 등이 분포하는 것을 볼 수 있는데. 막대 그래프를 보면 지가가 낮은 3개 구가 높은 3개 구보다 더 높은 비중을 차지하는 것을 볼 수 있음."
                    "위의 자료로 work와 공시지가와 상관관계 분석을 해보았더니 상관계수 -0.46으로 이전의 양의 상관관계와는 다른 중간정도 음의 상관관계를 확인할 수 있었고, 이는 낮은 공시지가가 타지역으로 통근하는 사람수에 중간정도의 음의 상관관계를 가진다는 것을 나타내는데 이는 주거적인 측면의 공간적 불평등을 초래한다는 결과를 뒷받침 할 수 있다고 판단하였음."),
             ]),
            
        dcc.Tab(label='서론 및 결론', children=[
                html.H1("연구배경 및 서론"),
                html.P("불평등은 빈곤퇴치를 포함한 인류의 후생과 관련된 긴밀한 문제로 포용 가능하고 지속가능한 정책 및 계획을 수립하는 데에 있어서 끊임없이 연구되어야 할 문제이고 산출방식에 따라 측정방법이 다양하게 존재함.\
                       이번 분석에서 다뤄볼 불평등은 공간 불평등으로 공간적인 범위는 서울시 내에 25개의 행정적 시군 구로 설정하였고, 크게 주거, 교통, 문화적인 측면에서 지역별 불평등이 존재할 것이라고 예상하고 분석함.\
                       이를 시사하는 방식으로는 대표적으로 서울시 시군 구별 공시지가와 다중 요인들 간의 상관관계를 분석하는 방식으로 진행. 상관관계 분석의 종속 변수는 서울시 시군 구별 공시지가로 설정, 독립변수는 스타벅스 분포 밀도, 지하철 분포 밀도, 타 시군 구로 통근하는 사람들의 밀도로 설정하였음.\
                       상관관계 분석 전 종속변수인 공시지가에 대한 설명: 공시지가에 구별 평균가격을 내림차순으로 정리한 그래프. 공시지가에 상위 3개 구는 강남구, 중구, 서초구 순이고 하위 3개는 도봉구, 노원구, 강북구 순으로 정리됨.\
                       그리고 독립변수들의 단위는 모두 km^2당 개수 혹은 명수인 밀도로 통일하고 시작함.")
,
                html.H1("결론"),
                html.P("이 분석에서는 서울시 내 공간적 불평등을 공시지가와 여러 요인들의 상관관계를 보여주는 지표들과의 관계를 주거, 문화, 교통 등의 인프라 측면에서 분석한 내용을 담고 있음. <br>\
                        이 분석의 의의는 인프라가 잘 형성된 곳, 즉 인프라의 접근성이 좋은 위치에 지가가 높은 것으로 나타남. 역세권 중에서도 초역세권, 트리플 역세권과 스타벅스의 입점을 보여주는 스세권이라는 단어가 집근처의 편의시설을 우선시하면서 탄생한 신조어로 인프라가 좋지 않은 지역에서 역세권이라는 신조어는 불평등을 나타내는 또 다른 단어로 인식될 수 있음. 또한, 인프라와 더불어 통근자를 통한 주거환경에서도 높은 지가에 대한 주거환경과 업무환경의 접근성에 대한 공간적 불평등을 보여줄 수 있음. <br>\
                        이러한 분석들에 대한 결과는 공간적 불평등에 대한 여러 요인들 중 단면적인 결과이며, 추후 더 구체적인 분석을 통한 결과 도출의 분석이 필요로 함."),
                html.H1("데이터 출처"),
                html.P("스타벅스데이터 : 지방행정인허가데이터"),
                html.P("지하철 : 국가교통 데이터 오픈마켓"),
                html.P("타지역 통근자 : KOSIS 국가통계포털유동인구 : 서울시 열린데이터광장"),
                html.P("공시지가 : 서울시 열린데이터광장")
            ]),
        
    ])
])
#여기까지 통근자 레이아웃

# Callbacks

#starbucks callback

@app.callback(
    Output('choropleth-map', 'figure'),  # 수정된 부분: 'choropleth-chart' -> 'choropleth-map'
    [Input('choropleth-dropdown-starbucks', 'value')]
)
def update_choropleth_map(selected_item):
    if selected_item == 'Starbucks':
        geo_df = gdf_cafe
        geo_column = 'STAR densi'
    
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
        labels={geo_column: f'스타벅스 밀도'}
    )
    fig.update_layout(title=f'스타벅스 분포 밀도 단계 구분도')
    return fig

@app.callback(
    Output('bar-chart', 'figure'),
    [Input('choropleth-dropdown-starbucks', 'value')]
)
def update_bar_chart(selected_item):
    # 예시: gdf_cafe에 대한 로직
    # 'STAR densi'를 이용하여 바 차트 생성
    if selected_item == 'Starbucks':
        # 'Gangnam-gu', 'Seocho-gu', 'Jung-gu'에 대한 바 차트
        selected_districts = ['중구', '강남구', '서초구', '노원구', '도봉구', '강북구']
        data1 = gdf_cafe[gdf_cafe['SIG_KOR_NM'].isin(selected_districts)].copy()

        # x축 순서를 조절하기 위해 selected_districts 순서로 정렬
        data1['SIG_KOR_NM'] = pd.Categorical(data1['SIG_KOR_NM'], categories=selected_districts, ordered=True)
        data1 = data1.sort_values('SIG_KOR_NM')

        fig1 = px.bar(
            data1, 
            x='SIG_KOR_NM', 
            y='STAR densi', 
            title='상하위 3개구',
            height=500, 
            color='STAR densi', 
            color_continuous_scale='Viridis',
            labels={'SIG_KOR_NM': '시군구','STAR densi': '스타벅스 밀도'}
        )

        return fig1
    else:
        # 기본값 또는 다른 경우 처리
        return {}, {}
    
@app.callback(
    Output('correlation-graph1', 'figure'),
    [Input('choropleth-dropdown-starbucks', 'value')]
)
def update_correlation_graph(selected_item):
    if selected_item == 'Starbucks':
        x_data = gdf_cafe['STAR densi']
    correlation_coefficient = np.corrcoef(gdf_price['land_price'], x_data)[0, 1]

    fig = px.scatter(x=x_data, y=gdf_price['land_price'], labels={'x': f'스타벅스 밀도', 'y': '공시지가'}, trendline="ols")
   
    fig.update_layout(
        title=f'스타벅스 분포 밀도와 공시지가의 상관관계',
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
    Output('correlation-graph2', 'figure'),
    [Input('choropleth-dropdown-starbucks', 'value')]
)
def update_correlation_graph(selected_item):
    if selected_item == 'Starbucks':
        x_data = gdf_cafe['BBAEK dens']
    correlation_coefficient = np.corrcoef(gdf_price['land_price'], x_data)[0, 1]

    fig = px.scatter(x=x_data, y=gdf_price['land_price'], labels={'x': f'빽다방 밀도', 'y': '공시지가'}, trendline="ols")
   
    fig.update_layout(
        title='빽다방 분포 밀도와 공시지가의 상관관계',
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

#starbucks callback

@app.callback(
    Output('choropleth-map_sub', 'figure'),  # 수정된 부분: 'choropleth-chart' -> 'choropleth-map'
    [Input('choropleth-dropdown-sub', 'value')]
)
def update_choropleth_map(selected_item):
    if selected_item == 'Subway':
        geo_df = gdf_sub
        geo_column = 'sub_densit'
    
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
        labels={geo_column: f'지하철역 밀도'}
    )
    fig.update_layout(title=f'지하철역 분포 밀도 단계 구분도')
    return fig

@app.callback(
    Output('bar-chart_sub', 'figure'),
    [Input('choropleth-dropdown-sub', 'value')]
)
def update_bar_chart(selected_item):
    # 예시: gdf_cafe에 대한 로직
    # 'STAR densi'를 이용하여 바 차트 생성
    if selected_item == 'Subway':
        # 'Gangnam-gu', 'Seocho-gu', 'Jung-gu'에 대한 바 차트
        selected_districts = ['중구', '강남구', '노원구', '서초구', '도봉구', '강북구']
        data1 = gdf_sub[gdf_sub['SIG_KOR_NM'].isin(selected_districts)].copy()

        # x축 순서를 조절하기 위해 selected_districts 순서로 정렬
        data1['SIG_KOR_NM'] = pd.Categorical(data1['SIG_KOR_NM'], categories=selected_districts, ordered=True)
        data1 = data1.sort_values('SIG_KOR_NM')

        fig1 = px.bar(
            data1, 
            x='SIG_KOR_NM', 
            y='sub_densit', 
            title='상하위 3개구',
            height=500, 
            color='sub_densit', 
            color_continuous_scale='Viridis',
            labels={'SIG_KOR_NM': '시군구','sub_densit': '지하철 밀도'}
        )

        return fig1
    else:
        # 기본값 또는 다른 경우 처리
        return {}, {}
    
@app.callback(
    Output('correlation-graph_sub', 'figure'),
    [Input('choropleth-dropdown-sub', 'value')]
)
def update_correlation_graph(selected_item):
    if selected_item == 'Subway':
        x_data = gdf_sub['sub_densit']
    correlation_coefficient = np.corrcoef(gdf_price['land_price'], x_data)[0, 1]

    fig = px.scatter(x=x_data, y=gdf_price['land_price'], labels={'x': f'지하철 밀도', 'y': '공시지가'}, trendline="ols")
   
    fig.update_layout(
        title=f'지하철역의 분포밀도와 공시지가의 상관관계',
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
    Output('correlation-graph_sub2', 'figure'),
    [Input('choropleth-dropdown-sub', 'value')]
)
def update_correlation_graph(selected_item):
    if selected_item == 'Subway':
        x_data = gdf_sub['bus_densit']
    correlation_coefficient = np.corrcoef(gdf_price['land_price'], x_data)[0, 1]

    fig = px.scatter(x=x_data, y=gdf_price['land_price'], labels={'x': f'버스정류장 밀도', 'y': '공시지가'}, trendline="ols")
   
    fig.update_layout(
        title=f'버스정류장 분포 밀도와 공시지가의 상관관계',
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
#subway tap callback

@app.callback(
    Output('choropleth-map_work', 'figure'),  # 수정된 부분: 'choropleth-chart' -> 'choropleth-map'
    [Input('choropleth-dropdown-work', 'value')]
)
def update_choropleth_map(selected_item):
    if selected_item == 'Work':
        geo_df = gdf_work
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
        labels={geo_column: f'통근자 밀도'}
    )
    fig.update_layout(title=f'타지역 통근자 분포 밀도 단계 구분도')
    return fig

@app.callback(
    Output('bar-chart_work', 'figure'),
    [Input('choropleth-dropdown-work', 'value')]
)
def update_bar_chart(selected_item):
    # 예시: gdf_cafe에 대한 로직
    # 'STAR densi'를 이용하여 바 차트 생성
    if selected_item == 'Work':
        # 'Gangnam-gu', 'Seocho-gu', 'Jung-gu'에 대한 바 차트
        selected_districts = ['도봉구', '노원구', '강북구', '중구', '강남구', '서초구']
        data1 = gdf_work[gdf_work['SIG_KOR_NM'].isin(selected_districts)].copy()

        # x축 순서를 조절하기 위해 selected_districts 순서로 정렬
        data1['SIG_KOR_NM'] = pd.Categorical(data1['SIG_KOR_NM'], categories=selected_districts, ordered=True)
        data1 = data1.sort_values('SIG_KOR_NM')

        fig1 = px.bar(
            data1, 
            x='SIG_KOR_NM', 
            y='density', 
            title='상하위 3개구',
            height=500, 
            color='density', 
            color_continuous_scale='Viridis',
            labels={'SIG_KOR_NM': '시군구','density': '통근자 밀도'}
        )

        return fig1
    else:
        # 기본값 또는 다른 경우 처리
        return {}, {}
    
@app.callback(
    Output('correlation-graph_work', 'figure'),
    [Input('choropleth-dropdown-work', 'value')]
)
def update_correlation_graph(selected_item):
    if selected_item == 'Work':
        x_data = gdf_work['density']
    correlation_coefficient = np.corrcoef(gdf_price['land_price'], x_data)[0, 1]

    fig = px.scatter(x=x_data, y=gdf_price['land_price'], labels={'x': f'통근자 밀도', 'y': '공시지가'}, trendline="ols")
   
    fig.update_layout(
        title=f'타지역 통근자의 밀도와 공시지가의 상관관계',
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

#Work tap call

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)


# In[ ]:




