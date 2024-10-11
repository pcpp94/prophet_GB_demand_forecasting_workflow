import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go

# Initialize the Dash app
app = dash.Dash(__name__)

# Time and orders
n_orders = 2
orders = np.linspace(1, n_orders, n_orders)
t = np.linspace(0, 360, 361) / 360

# Define layout for app
app.layout = html.Div([
    dcc.Graph(id='graph'),  # First graph for individual sin and cos functions
    dcc.Graph(id='sum_graph'),  # Second graph for the sum of all

    # Sliders for amplitude control
    dcc.Slider(
        id='amp_sin', min=0, max=2, step=0.1, value=0.2,
        marks={i: f'{i}' for i in range(3)},
        tooltip={"placement": "bottom", "always_visible": True},
        included=True
    ),
    dcc.Slider(
        id='amp_cos', min=0, max=2, step=0.1, value=0.4,
        marks={i: f'{i}' for i in range(3)},
        tooltip={"placement": "bottom", "always_visible": True},
        included=True
    ),
    dcc.Slider(
        id='amp_sin2', min=0, max=2, step=0.1, value=0.2,
        marks={i: f'{i}' for i in range(3)},
        tooltip={"placement": "bottom", "always_visible": True},
        included=True
    ),
    dcc.Slider(
        id='amp_cos2', min=0, max=2, step=0.1, value=0.4,
        marks={i: f'{i}' for i in range(3)},
        tooltip={"placement": "bottom", "always_visible": True},
        included=True
    ),
])

# Callback to update graphs dynamically


@app.callback(
    [Output('graph', 'figure'),
     Output('sum_graph', 'figure')],
    [Input('amp_sin', 'value'),
     Input('amp_cos', 'value'),
     Input('amp_sin2', 'value'),
     Input('amp_cos2', 'value')])
def update_graph(amp_sin, amp_cos, amp_sin2, amp_cos2):
    # Create the sine and cosine functions
    traces = []
    total_sum = np.zeros_like(t)  # Initialize the sum graph

    amp_s = dict(zip([1, 2], [amp_sin, amp_sin2]))
    amp_c = dict(zip([1, 2], [amp_cos, amp_cos2]))

    for order in orders:
        y_sin = amp_s[order] * np.sin(2 * np.pi * order * t)
        y_cos = amp_c[order] * np.cos(2 * np.pi * order * t)

        # Add individual sin and cos traces to the main graph
        traces.append(go.Scatter(x=t, y=y_sin, mode='lines',
                      name=f'Sin (Order {order})'))
        traces.append(go.Scatter(x=t, y=y_cos, mode='lines',
                      name=f'Cos (Order {order})'))

        # Add the current sine and cosine functions to the sum
        total_sum += y_sin + y_cos

    # Define layout for the individual sine and cosine plot
    layout = go.Layout(
        title='Dynamic Sine and Cosine Plot',
        xaxis={'title': 't'},
        yaxis={'title': 'Amplitude'},
    )

    # Define layout for the sum graph
    sum_layout = go.Layout(
        title='Sum of Sines and Cosines',
        xaxis={'title': 't'},
        yaxis={'title': 'Sum Amplitude'},
    )

    # Create sum trace
    sum_trace = go.Scatter(x=t, y=total_sum, mode='lines', name='Sum')

    # Return both figures
    return {'data': traces, 'layout': layout}, {'data': [sum_trace], 'layout': sum_layout}


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
