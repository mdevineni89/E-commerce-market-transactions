import dash
import dash_table
import base64
import datetime
from datetime import date
import io
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import plotly.graph_objects as go
# import dash_table
from dash.dependencies import Input, Output,State
import numpy as np
import pandas as pd
import plotly.figure_factory as ff


new_dataset="Online-Retail-cleaned.csv"

df=pd.read_csv(new_dataset)
pie_df=df.groupby('Day of week')['FinalPrice'].count()
orders = df.groupby(by=['CustomerID','Country','InvoiceMonth','Day of week'], as_index=False)['InvoiceNo'].count()
df_numeric=df[['UnitPrice','Quantity','FinalPrice']]
heatmap_df = df.pivot_table(index = 'InvoiceMonth',columns = 'Day of week', values = 'FinalPrice', aggfunc='sum')
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
colors = {
    'background': '#111111',
    'text': 'white'
}
all_continents = df['Continent'].unique()
all_days=df['Day of week'].unique()



# styling the sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "linen",


}

# padding for the page content
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("MENU", className="display-4"),
        html.Hr(),
        html.P(
            "E-commerce transactions ", className="lead",style={'fontcolor':'lime'}
        ),
        dbc.Nav(
            [
                dbc.NavLink("Distribution Plot", href="/", active="exact"),
                dbc.NavLink("Pie chart", href="/page-1", active="exact"),
                dbc.NavLink("Bar chart", href="/page-2", active="exact"),
                dbc.NavLink("line plot", href="/page-3", active="exact"),
                dbc.NavLink("Scatter plot", href="/page-4", active="exact"),
                dbc.NavLink("Heatmap", href="/page-5", active="exact"),
                dbc.NavLink("Histogram Plot", href="/page-6", active="exact"),
                dbc.NavLink("Count Plot", href="/page-7", active="exact"),
                dbc.NavLink("Box Plot", href="/page-8", active="exact"),
                dbc.NavLink("Pair Plot", href="/page-9", active="exact"),
                dbc.NavLink("Violin Plot", href="/page-10", active="exact"),
                dbc.NavLink("Bar(group) Plot", href="/page-11", active="exact"),
                dbc.NavLink("Area Plot", href="/page-12", active="exact"),
                dbc.NavLink("KDE Plot", href="/page-13", active="exact"),
                dbc.NavLink("Download component", href="/page-14", active="exact"),
                dbc.NavLink("Upload component", href="/page-15", active="exact"),


            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)

app.layout = html.Div(style={'backgroundColor':'azure'},children=[
             html.H1(

             style={
            'textAlign': 'center',
            'color': colors['text']
                }),

    dcc.Location(id="url"),

    sidebar,
    content
])


dis_layout = html.Div([html.H1('Unitprice vs Quantity', style={'textAlign':'center'}),
                       html.H1(

             style={
            'textAlign': 'center',
            'color': colors['text']
                }),
    dcc.Graph(id="graph"),
    html.P("Select Distribution:"),
    dcc.RadioItems(
        id='dist-marginal',
        options=[{'label': x, 'value': x}
                 for x in ['box', 'violin','rug']],
        value='box'
    )
])

@app.callback(
    Output("graph", "figure"),
    [Input("dist-marginal", "value")])

def display_graph(marginal):
    fig = px.histogram(
        df[df.FinalPrice>0], x="Quantity", y="UnitPrice",
        marginal=marginal, range_x=[-5, 60],
        hover_data=df.columns)

    return fig

pie_layout = html.Div([
    html.H1('Final Price on respective Day of week'),
html.H1(

             style={
            'textAlign': 'center',
            'color': colors['text']
                }),
    dcc.Graph(id='pie-dropdown'),
    dcc.Input(id='pie_chart',value='pie')
])
@app.callback(Output('pie-dropdown', 'figure'),
              Input('pie_chart','value'))

def pie_dropdown(value1):
    fig = px.pie(values=pie_df.values, names=pie_df.index)
    return fig


bar_layout = html.Div([html.H1('Number of Orders', style={'textAlign':'center'}),
html.H1(

             style={
            'textAlign': 'center',
            'color': colors['text']
                }),
                         dcc.Graph(id="my_graph"),
                         dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': 'Month', 'value': 'Month'},
                {'label': 'Day of Week', 'value': 'Day of Week'},
                    ],
            value='Month'),
            html.H1(["choose option:"]),
            dcc.Dropdown(
            id='con_dropdown',
            options=[
                {'label': 'United Kingdom', 'value': 'United Kingdom'},
                {'label': 'Iceland', 'value': 'Iceland'},
                {'label': 'France', 'value': 'France'},
                {'label': 'Australia', 'value': 'Australia'},
                {'label': 'Netherlands', 'value': 'Netherlands'},
                {'label': 'Germany', 'value': 'Germany'},
                {'label': 'Norway', 'value': 'Norway'},
                {'label': 'EIRE', 'value': 'EIRE'},
                {'label': 'Switzerland', 'value': 'Switzerland'},
                {'label': 'Spain', 'value': 'Spain'},
                {'label': 'Poland', 'value': 'Poland'},
                {'label': 'Portugal', 'value': 'Portugal'},
                {'label': 'Italy', 'value': 'Italy'},
                {'label': 'Belgium', 'value': 'Belgium'},
                {'label': 'Lithuania', 'value': 'Lithuania'},
                {'label': 'Japan', 'value': 'Japan'},
                {'label': 'Channel Islands', 'value': 'Channel Islands'},
                {'label': 'Denmark', 'value': 'Denmark'},
                {'label': 'Cyprus', 'value': 'Cyprus'},
                {'label': 'Sweden', 'value': 'Sweden'},
                {'label': 'Finland', 'value': 'Finland'},
                {'label': 'Austria', 'value': 'Austria'},
                {'label': 'Greece', 'value': 'Greece'},
                {'label': 'Singapore', 'value': 'Singapore'},
                {'label': 'Lebanon', 'value': 'Lebanon'},
                {'label': 'United Arab Emirates', 'value': 'United Arab Emirates'},
                {'label': 'Israel', 'value': 'Israel'},
                {'label': 'Saudi Arabia', 'value': 'Saudi Arabia'},
                {'label': 'Czech Republic', 'value': 'Czech Republic'},
                {'label': 'Canada', 'value': 'Canada'},
                {'label': 'Brazil', 'value': 'Brazil'},
                {'label': 'USA', 'value': 'USA'},
                {'label': 'European Community', 'value': 'European Community'},
                {'label': 'Bahrain', 'value': 'Bahrain'},
                {'label': 'Malta', 'value': 'Malta'},
                {'label': 'RSA', 'value': 'RSA'}
                    ],
            value='UK',

            multi=True,

            ),




 ])

@app.callback(
     Output(component_id='my_graph', component_property='figure'),
     [Input(component_id='dropdown', component_property='value'),
      Input(component_id='con_dropdown',component_property='value')]

)
def select_graph(value1, value2):

        col=value2
        flag=0

        for i in col:
            if flag==0:
                select = orders[orders['Country'] == i]
                flag+=1
            else:
                select = select.append(orders[orders['Country'] == i])
        if value1 == 'Month':
            fig = px.bar(data_frame=select,x='InvoiceMonth', y='InvoiceNo',color='Country')
            return fig
        else:
            fig = px.bar(data_frame=select, x='Day of week', y='InvoiceNo', color='Country')
            return fig

line_layout= html.Div([html.H1('UnitPrice vs Quantity'),
html.H1(

             style={
            'textAlign': 'center',
            'color': colors['text']
                }),
            dcc.Checklist(
            id="checklist",
            options=[{"label": x, "value": x}
                    for x in all_continents],
            value=all_continents[3:],
            labelStyle={'display': 'inline-block'}
    ),
    dcc.Graph(id="line-chart"),
])

@app.callback(
    Output("line-chart", "figure"),
    [Input("checklist", "value")])
def update_line_chart(continents):
    mask = df.Continent.isin(continents)
    fig = px.line(df[mask],
        x='Day of week',y="UnitPrice", color='Country')
    return fig

scatter_layout= html.Div([html.H1('UnitPrice vs Quantity'),

    dcc.Graph(id="scatter-plot"),
dcc.Input(
        id='scatter_id',
        value='scatterplot'

    ),
    dcc.DatePickerRange(
                              id='my-date-picker-range',
                              min_date_allowed=date(2010, 1, 5),
                              max_date_allowed=date(2011, 9, 19),
                              initial_visible_month=date(2011, 8, 5),
                              end_date=date(2011, 12, 25)
                         ),
                          # html.Div(id='output-container-date-picker-range')
])

@app.callback(
    Output("scatter-plot", "figure"),
    [Input('scatter_id', 'value')])


def update_scatter_chart(value1):

    # low, high = slider_range
    # mask = (df['Quantity'] > low) & (df['Quantity'] < high)
    fig = px.scatter(
        df, x="Quantity", y="UnitPrice",
        color="Continent",
        hover_data=['Quantity'],trendline='ols')
    return fig

# upload_layout = html.Div([
#     dcc.Upload(html.Button('Upload File')),
#
#     html.Hr(),
#
#     dcc.Upload(html.A('Upload File')),
#
#     html.Hr(),
#
#     dcc.Upload([
#         'Drag and Drop or ',
#         html.A('Select a File')
#     ], style={
#         'width': '100%',
#         'height': '60px',
#         'lineHeight': '60px',
#         'borderWidth': '1px',
#         'borderStyle': 'dashed',
#         'borderRadius': '5px',
#         'textAlign': 'center'
#     })
# ])

heatmap_layout=html.Div([
    html.P("Day of the week:"),
    dcc.Input(
        id='days',
        value='heatmap'

    ),
    dcc.Graph(id="heat_graph"),
])

@app.callback(
    Output("heat_graph", "figure"),
    [Input("days", "value")])
def filter_heatmap(cols):
    fig = px.imshow(heatmap_df)
    return fig

hist_layout = html.Div([html.H1("Histogram Plot"),
    dcc.Graph(id="hist_graph"),
    html.P('select a variable'),
    dcc.Dropdown(
        id='select',
        options=[
            {'label':'Quantity','value':'Quantity'},
            {'label':'UnitPrice','value':'UnitPrice'},
            {'label':'FinalPrice','value':'FinalPrice'},

        ],value='Quantity'
    ),
    html.P("No of bins:"),
    dcc.Slider(id='bins', min=10, max=100, value=10,
               marks= {10:'10',20:'20',30:'30',40:'40',50:'50',60:'60',70:'70',80:'80',90:'90',100:'100'}),

])

@app.callback(
    Output("hist_graph", "figure"),
    [Input("select", "value"),
     Input("bins","value")])

def display_hist(value2,bin):
    fig = px.histogram(df, nbins=int(bin),x=value2)
    return fig

count_layout=html.Div([
    html.H1('Count Plot', style={'textAlign':'center'}),
html.H1(

             style={
            'textAlign': 'center',
            'color': colors['text']
                }),
    html.Br(),
    dcc.Graph(id = 'count_graph'),
    html.P('select a variable'),
    dcc.Dropdown(
        id='select_var',
        options=[
            {'label':'Description','value':'Description'},
            {'label':'Continent','value':'Continent'},
            {'label':'Country','value':'Country'},


        ],value='Description'
    ),

])

@app.callback(
    Output(component_id='count_graph', component_property='figure'),
    Input(component_id='select_var',component_property='value'),
)

def display_count(value1):
    if value1=='Description':
        value_x=df[value1].unique()
        count_x=[len(df[df[value1]==i]) for i in value_x]
        d1={'valx':value_x,'c_x':count_x}
        d1_f=pd.DataFrame(d1)
        d2_f=d1_f.sort_values(by='c_x',ascending=False)
        fig1 = px.bar(d2_f.iloc[:15,:],x='valx', y='c_x', color='valx')
    else:
        value_x = df[value1].unique()
        count_x = [len(df[df[value1] == i]) for i in value_x]
        fig1 = px.bar(x=value_x, y=count_x, color=value_x)
    return fig1

box_layout=html.Div([html.H1("Box Plot",style={'textAlign':'center'}),
    html.P("x-axis:"),

    dcc.Checklist(
        id='x-axis',
        options=[{'value': x, 'label': x}
                 for x in ['Country', 'Continent', 'Day of Week', 'Description']],
        value=['Continent'],
        labelStyle={'display': 'inline-block'}
    ),
    html.P("y-axis:"),
    dcc.RadioItems(
        id='y-axis',
        options=[{'value': x, 'label': x}
                 for x in ['UnitPrice', 'Quantity', 'FinalPrice']],
        value='Quantity',
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Graph(id="box-plot"),
])

@app.callback(
    Output("box-plot", "figure"),
    [Input("x-axis", "value"),
     Input("y-axis", "value")])
def generate_chart(x, y):
    fig = px.box(df, x=x, y=y)
    return fig

pair_layout= html.Div([html.H1('Pair Plot'),
html.H1(

             style={
            'textAlign': 'center',
            'color': colors['text']
                }),
    dcc.Graph(id="pair-plot"),

    dcc.Input(
        id='pair',
        value='pair plot'
    ),
])

@app.callback(
    Output("pair-plot", "figure"),
    [Input("pair", "value")])
def update_pair(value1):

    fig = px.scatter_matrix(
        df,dimensions=['InvoiceNo','Quantity','UnitPrice','FinalPrice'],
        color="Continent")
    return fig

violin_layout=html.Div([html.H1('Violin Plot'),
html.H1(

             style={
            'textAlign': 'center',
            'color': colors['text']
                }),
    dcc.Graph(id="violin-plot"),

    dcc.Input(
        id='violin',
        value='violin plot'
    ),
])

@app.callback(
    Output("violin-plot", "figure"),
    [Input("violin", "value")])
def update_pair(value1):
    fig = px.violin(
        df,y='FinalPrice',points=False)
    return fig

df02=df[:5000]
bargroup_layout=html.Div([html.H1("Continent wise Unit price vs quantity graph on respective days ")

             ,
    dcc.Dropdown(
        id="bar_dropdown",
        options=[{"label": x, "value": x} for x in all_days],
        value=all_days[0],
        clearable=False,
    ),
    dcc.Graph(id="barg-chart"),
])

@app.callback(
    Output("barg-chart", "figure"),
    [Input("bar_dropdown", "value")])
def update_barg_chart(day):
    mask = df02[df02["Day of week"] == day]
    fig = px.bar(data_frame=mask, x="Quantity", y="UnitPrice",
                 color="Continent",barmode='group')
    return fig


area_layout=html.Div([html.H1("Area Plot",style={'textAlign':'center'}),
    html.P("Select y-axis"),

    dcc.Dropdown(
        id='y_axis',
        options=[
            {'label': x, 'value': x}
            for x in ['Quantity','UnitPrice', 'FinalPrice']],
        value='UnitPrice'
    ),
    dcc.Graph(id="area_graph"),
])

@app.callback(
    Output("area_graph", "figure"),
    [Input("y_axis", "value")])
def display_area(y1):
    fig = px.area(
        df, x='Day of week',y=y1,
        color="Country", line_group="Country")
    return fig


df20=df[:50000]

kde_layout=html.Div([
    html.H1('KDE Plot', style={'textAlign':'center'}),
html.H1(

             style={
            'textAlign': 'center',
            'color': colors['text']
                }),
    dcc.Graph(id='kde_graph'),
    html.P('select variable'),
    dcc.Dropdown(
        id='kde_var',
        options=[
            {'label':'Quantity','value':'Quantity'},
            {'label':'UnitPrice','value':'UnitPrice'},
            {'label':'FinalPrice','value':'FinalPrice'},

        ],value='FinalPrice',
    ),


])

@app.callback(
    Output(component_id='kde_graph', component_property='figure'),
    [Input(component_id='kde_var',component_property='value'),]
)

def display_kde(value1):
    group_labels = [value1]
    varr1=np.array(df20[value1])
    varr1=varr1.flatten()
    varr1=[varr1]
    fig = ff.create_distplot(varr1,group_labels)
    return fig

download_layout = dbc.Container(
    [
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns[:100]],
            data=df.to_dict('records'),
        ),

        dbc.Button(id='btn',
            children=[html.I(className="fa fa-download mr-1"), "Download"],
            color="info",
            className="mt-1"
        ),
        dcc.Textarea(
            placeholder='INPUT YOUR FEEDBACK',
            value="INPUT YOUR FEEDBACK",
            style={'width':"80%"}
    ),

        dcc.Download(id="download-component"),
    ],
    className='m-4'
)



@app.callback(
    Output("download-component", "data"),
    Input("btn", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dict(content="Always remember, we're better together.", filename="hello.txt")

upload_layout=html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-div'),
    html.Div(id='output-datatable'),
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        html.P("Inset X axis data"),
        dcc.Dropdown(id='xaxis-data',
                     options=[{'label':x, 'value':x} for x in df.columns[:100]]),
        html.P("Inset Y axis data"),
        dcc.Dropdown(id='yaxis-data',
                     options=[{'label':x, 'value':x} for x in df.columns[:100]]),
        html.Button(id="submit-button", children="Create Graph"),
        html.Hr(),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns[:100]],
            page_size=15
        ),
        dcc.Store(id='stored-data', data=df.to_dict('records')),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


@app.callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


@app.callback(Output('output-div', 'children'),
              Input('submit-button','n_clicks'),
              State('stored-data','data'),
              State('xaxis-data','value'),
              State('yaxis-data', 'value'))
def make_graphs(n, data, x_data, y_data):
    if n is None:
        return dash.no_update
    else:
        bar_fig = px.bar(data, x=x_data, y=y_data)
        # print(data)
        return dcc.Graph(figure=bar_fig)



@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return dis_layout

    elif pathname == "/page-1":
        return pie_layout
    elif pathname == "/page-2":
        return bar_layout
    elif pathname == "/page-3":
        return line_layout
    elif pathname == "/page-4":
        return scatter_layout

    elif pathname == "/page-5":
        return heatmap_layout
    elif pathname == "/page-6":
        return hist_layout
    elif pathname == "/page-7":
        return count_layout
    elif pathname == "/page-8":
        return box_layout
    elif pathname == "/page-9":
        return pair_layout
    elif pathname == "/page-10":
        return violin_layout
    elif pathname == "/page-11":
        return bargroup_layout
    elif pathname == "/page-12":
        return area_layout
    elif pathname == "/page-13":
        return kde_layout
    elif pathname == "/page-14":
        return download_layout
    elif pathname == "/page-15":
        return upload_layout

    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__=='__main__':
    app.run_server(debug=True, port=3068)