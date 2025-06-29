import dash
from dash import Dash, dcc, html, callback, Input, Output, State
import base64
import io
import gzip
from Bio import SeqIO
import plotly.graph_objs as go

app = Dash(suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Div([
        html.Img(src='assets/spot logo.png', style={
            'height': '110px',
            'width': 'auto',
            'maxWidth': '220px',
            'margin': '0 auto',
            'display': 'block',
        }),
        html.Div(
            'Early plant disease diagnosis and precision farming for sustainable agriculture.',
            style={
                'textAlign': 'center',
                'fontFamily': 'Segoe UI, Arial, sans-serif',
                'fontSize': '20px',
                'color': '#2c3e50',
                'marginTop': '18px',
                'marginBottom': '0',
                'fontStyle': 'italic',
            }
        )
    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'justifyContent': 'center',
        'marginBottom': '30px',
    }),
    # represents the browser address bar and doesn't render anything
    dcc.Location(id='url', refresh=False),

    html.Div([
        html.Button('Search Sequence(s)', id='btn-home', n_clicks=0, style={
            'marginRight': '16px',
            'padding': '12px 32px',
            'border': 'none',
            'borderRadius': '8px',
            'background': 'rgb(105, 138, 70)',
            'color': 'white',
            'fontSize': '18px',
            'fontWeight': '600',
            'boxShadow': '0 2px 8px rgba(79,140,255,0.15)',
            'cursor': 'pointer',
            'transition': 'background 0.3s',
        }),
        html.Button('About', id='btn-about', n_clicks=0, style={
            'padding': '12px 32px',
            'border': 'none',
            'borderRadius': '8px',
            'background': 'rgb(105, 138, 70)',
            'color': 'white',
            'fontSize': '18px',
            'fontWeight': '600',
            'boxShadow': '0 2px 8px rgba(255,126,95,0.15)',
            'cursor': 'pointer',
            'transition': 'background 0.3s',
        })
    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center', 'marginBottom': '32px'}),

    # content will be rendered in this element
    html.Div(id='page-content')
], style={'minHeight': '100vh', 'background': '#f5f5f5'})

@callback(Output('url', 'pathname'),
          [Input('btn-home', 'n_clicks'), Input('btn-about', 'n_clicks')],
          [State('url', 'pathname')])
def navigate(btn_home, btn_about, current_pathname):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_pathname
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'btn-home':
        return '/'
    elif button_id == 'btn-about':
        return '/about'
    return current_pathname


def dummy_plot_and_score():
    # Example data for fungus species and their match scores
    species_scores = [
        ("Fusarium oxysporum", 80.1),
        ("Botrytis cinerea", 76.5),
        ("Alternaria alternata", 68.9),
        ("Colletotrichum gloeosporioides", 55.2),
        ("Magnaporthe oryzae", 42.7),
    ]
    return html.Div([
        html.Div([
            html.Img(src='/assets/plot.png', style={'width': '100%', 'maxWidth': '420px', 'display': 'block', 'borderRadius': '12px', 'boxShadow': '0 2px 8px rgba(44,62,80,0.10)', 'marginRight': '32px'}),
            html.Img(src='/assets/plot2.png', style={'width': '100%', 'maxWidth': '420px', 'display': 'block', 'borderRadius': '12px', 'boxShadow': '0 2px 8px rgba(44,62,80,0.10)'}),
        ], style={
            'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center', 'alignItems': 'center', 'margin': '0 auto', 'marginBottom': '18px'
        }),
        html.Div([
            html.H4("Disease Match Scores", style={"marginBottom": "12px", "textAlign": "center"}),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Species", style={"padding": "8px 18px", "fontWeight": "bold", "fontSize": "18px", "background": "#e8f5e9", "textAlign": "left"}),
                        html.Th("Type", style={"padding": "8px 18px", "fontWeight": "bold", "fontSize": "18px", "background": "#e8f5e9", "textAlign": "left"}),
                        html.Th("Match Score (%)", style={"padding": "8px 18px", "fontWeight": "bold", "fontSize": "18px", "background": "#e8f5e9", "textAlign": "right"})
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(species, style={"padding": "8px 18px", "fontSize": "17px", "textAlign": "left"}),
                        html.Td("Fungus", style={"padding": "8px 18px", "fontSize": "17px", "textAlign": "left"}),
                        html.Td(f"{score}", style={"padding": "8px 18px", "fontSize": "17px", "textAlign": "right"})
                    ]) for species, score in species_scores
                ])
            ], style={"width": "100%", "maxWidth": "650px", "margin": "0 auto", "borderCollapse": "collapse", "boxShadow": "0 2px 8px rgba(44,62,80,0.07)", "background": "#fafafa", "borderRadius": "8px", "overflow": "hidden"})
        ], style={"marginTop": "24px"})
    ])

def parse_fasta(text):
    return dummy_plot_and_score()

def parse_fna_gz(contents):
    return dummy_plot_and_score() 

@callback(
    Output('search-result', 'children'),
    Input('search-btn', 'n_clicks'),
    State('fasta-sequence', 'value'),
    State('upload-fna', 'contents'),
    prevent_initial_call=True
)

def search_sequences(n_clicks, fasta_text, upload_contents):
    if not n_clicks:
        return ''
    if upload_contents:
        try:
            return parse_fna_gz(upload_contents)
        except Exception as e:
            return html.Div([f"Error parsing FNA (FASTA): {e}",], style={"color": "red"})
    elif fasta_text:
        try:
            return parse_fasta(fasta_text)
        except Exception as e:
            return html.Div([f"Error parsing FASTA: {e}"], style={"color": "red"})
    else:
        return html.Div(["Please provide a FASTA sequence or upload a .fna.gz file."], style={"color": "red"})

@callback(
    Output('upload-message', 'children'),
    Input('upload-fna', 'filename'),
    Input('upload-fna', 'contents')
)
def show_upload_message(filename, contents):
    if filename and contents:
        return f"File '{filename}' uploaded successfully."
    return ''

@callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/about':
        return html.Div([
            html.H2('About'),
            html.H3('Early Disease Detection'),
            html.Ul([
                html.Li([
                    html.B('Rapid Diagnostics:'),
                    ' From in-field RNA sampling to AI-powered disease detection, we identify pathogens days to weeks before symptoms are visible.'
                ]),
                html.Li([
                    html.B('Drone & Satellite Analytics:'),
                    ' Multispectral, thermal and RGB imagery pinpoint emerging stress hotspots across entire fields.'
                ]),
                html.Li([
                    html.B('Predictive Modeling:'),
                    ' Weather, soil and topography layers feed machine learning models that forecast outbreaks and recommend targeted interventions.'
                ]),
            ]),
            html.H3('Precision Farming Solutions'),
            html.Ul([
                html.Li([
                    html.B('Variable Rate Application:'),
                    ' Precise prescriptions apply fungicides, nutrients or water only where needed, cutting costs and chemical use.'
                ]),
                html.Li([
                    html.B('IoT Sensor Networks:'),
                    ' Soil moisture, leaf wetness and microclimate sensors stream real-time data to our decision support dashboard.'
                ]),
                html.Li([
                    html.B('Digital Twin Fields:'),
                    ' 3D crop models simulate growth under different management scenarios, helping growers to plan and choose their ideal fields.'
                ]),
            ]),
            html.H3('Sustainable Outcomes'),
            html.Ul([
                html.Li([
                    html.B('Yield Protection:'),
                    ' Early action prevents losses that can exceed 30% in unmanaged disease outbreaks.'
                ]),
                html.Li([
                    html.B('Resource Stewardship:'),
                    ' Precision inputs reduce fertilizer runoff and pesticide load,protecting waterways and beneficial insects.'
                ]),
                html.Li([
                    html.B('Climate Resilience:'),
                    ' Healthier plants with optimized nutrition and water use tolerate heat and drought stress more effectively.'
                ]),
            ]),
            html.H3('Why It Matters'),
            html.P('Traditional spray and pray farming reacts to plant disease often at the mid to end stage, costing resources and accelerating resistance. Our proactive approach shifts the paradigm: detect, decide, and deliver exactly what each plant needs. The result is healthier crops, lower carbon footprints, and stronger food security.'),
            html.H3('Our Approach'),
            html.Ul([
                html.Li(html.B('Science First:')), html.Span(' Grounded in plant pathology, genomics, and agronomy.'),
                html.Li(html.B('Tech-Driven:')), html.Span(' Leveraging AI, remote sensing, and edge computing.'),
                html.Li(html.B('Farmer-Centric:')), html.Span(' Built with and for growers, agronomists, and crops.'),
                html.Li(html.B('Open & Collaborative:')), html.Span(' APIs and partnerships that integrate seamlessly with existing machinery and farm management platforms.'),
            ]),
            html.H3('Join Us'),
            html.P('Whether you are a producer striving for higher margins, a food company aiming to de-risk its supply chain, or a researcher pushing the boundaries of agricultural science, we invite you to collaborate. Together, we can cultivate healthier fields, healthier ecosystems, and a healthier planet.'),
            html.H3('Team'),
            html.Ul([
                html.Li('Molly Bergum'),
                html.Li('Ricardo Valencia'),
                html.Li('Yufei Li'),
                html.Li('Xiyao (Miranda) Shou'),
            ]),
            html.H4('Detect early. Act precisely. Harvest sustainably.'),
            html.P([
                'Contact us at ',
                html.A('info@spottearly.com', href='mailto:info@precision-diagnosis.ag'),
                ' to learn more or schedule a demo.'
            ]),
        ], style={'maxWidth': '800px', 'margin': '0 auto', 'fontFamily': 'Segoe UI, Arial, sans-serif', 'fontSize': '18px', 'lineHeight': '1.7', 'background': '#f5f5f5', 'padding': '32px', 'borderRadius': '12px'})
    else:
        return html.Div([
            html.H2('Search', style={'marginBottom': '0'}),
            html.Div('Match mDNA sequence with different types of disease', style={
                'textAlign': 'left',
                'fontSize': '16px',
                'color': '#2c3e50',
                'marginBottom': '30px',
                'marginTop': '0',
                'fontStyle': 'italic',
            }),
            html.Div('Insert your FASTA sequence:', style={
                'textAlign': 'center',
                'fontWeight': 'bold',
                'marginBottom': '8px',
                'marginTop': '8px',
            }),
            dcc.Textarea(
                id='fasta-sequence',
                placeholder='Enter FASTA sequence here...',
                style={'width': '100%', 'height': 120, 'marginBottom': '18px', 'marginTop': '8px', 'borderColor': '#1976d2', 'borderWidth': '2px'}
            ),
            html.Div('Or upload a .fna.gz file:', style={
                'textAlign': 'center',
                'fontWeight': 'bold',
                'marginBottom': '8px',
                'marginTop': '8px',
            }),
            dcc.Upload(
                id='upload-fna',
                children=html.Div([
                    html.Span('Drag and Drop or ', style={'fontStyle': 'italic', 'fontSize': '14px'}),
                    html.A('Select File', style={'fontStyle': 'italic', 'fontSize': '14px'})
                ]),
                accept='.fna.gz',
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '8px',
                    'textAlign': 'center',
                    'marginBottom': '18px',
                    'background': '#f9f9f9',
                    'borderColor': '#1976d2',
                },
                multiple=False
            ),
            html.Div(id='upload-message', style={'marginBottom': '8px', 'color': 'green', 'fontWeight': 'bold'}),
            html.Button('Search', id='search-btn', n_clicks=0, style={
                'width': '100%',
                'padding': '14px',
                'background': '#9cd371',
                'color': 'white',
                'fontSize': '18px',
                'fontWeight': '600',
                'border': 'none',
                'borderRadius': '8px',
                'cursor': 'pointer',
                'boxShadow': '0 2px 8px rgba(105,138,70,0.10)',
                'marginTop': '8px',
            }),
            html.Div(id='search-result', style={'marginTop': '18px'})
        ], style={'maxWidth': '800px', 'margin': '0 auto', 'fontFamily': 'Segoe UI, Arial, sans-serif', 'fontSize': '18px', 'lineHeight': '1.7', 'background': '#f5f5f5', 'padding': '32px', 'borderRadius': '12px'})

if __name__ == '__main__':
    app.run(debug=True)