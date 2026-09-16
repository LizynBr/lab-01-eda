import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from app import app, server
from pages import index, predictions


# Navebar for header of app
navbar = dbc.NavbarSimple(
    brand='Fish Dimensions Regression Analysis',
    brand_href='/', 
    children=[
        dbc.NavItem(dcc.Link('Predictions', href='/predictions', className='nav-link')), 
    ],
    sticky='top',
    color='light', 
    dark=False
)

# The footer at the bottom of the screen
footer = dbc.Container(
    dbc.Row(
        dbc.Col(
            html.P(
                [
                    html.Span('Tyler Russin', className='mr-2'),

                    # Twitter URL
                    html.A(html.I(className='fab fa-twitter mr-1'), href='https://twitter.com/tyler_russin', style={
                                                                                                                    "padding-top": "2px",
                                                                                                                    "padding-right": "2px",
                                                                                                                    "padding-bottom": "2px",
                                                                                                                    "padding-left": "2px",
                                                                                                                    "color": "rgb(17,157,255)"
                                                                                                                    }),
                    # Linkedin URL
                    html.A(html.I(className='fab fa-linkedin mr-1'), href='https://www.linkedin.com/in/tyler-russin/', style={
                                                                                                                            "padding-top": "2px",
                                                                                                                            "padding-right": "2px",
                                                                                                                            "padding-bottom": "2px",
                                                                                                                            "padding-left": "2px",
                                                                                                                            "color": "rgb(17,157,255)"
                                                                                                                            }),
                    # GitHub URL
                    html.A(html.I(className='fab fa-github-square mr-1'), href='https://github.com/tylerrussin/Fish-Dimensions-Regression-Analysis', style={
                                                                                                                                                            "padding-top": "2px",
                                                                                                                                                            "padding-right": "2px",
                                                                                                                                                            "padding-bottom": "2px",
                                                                                                                                                            "padding-left": "2px",
                                                                                                                                                            "color": "rgb(17,157,255)"
                                                                                                                                                            }),
                    # Personal Email    
                    html.A(html.I(className='fas fa-envelope-square mr-1'), href='mailto:tylerrussin2@gmail.com', style={
                                                                                                                        "padding-top": "2px",
                                                                                                                        "padding-right": "2px",
                                                                                                                        "padding-bottom": "2px",
                                                                                                                        "padding-left": "2px",
                                                                                                                        "color": "rgb(17,157,255)"
                                                                                                                        }), 


                ], 
                className='lead'
            )
        )
    )
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False), 
    navbar, 
    dbc.Container(id='page-content', className='mt-4'), 
    html.Hr(), 
    footer
])

# URL Routing for Multi-Page Apps
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return index.layout
    elif pathname == '/predictions':
        return predictions.layout
    else:
        return dcc.Markdown('## Page not found')

# Run app
if __name__ == '__main__':
    app.run(debug=True)