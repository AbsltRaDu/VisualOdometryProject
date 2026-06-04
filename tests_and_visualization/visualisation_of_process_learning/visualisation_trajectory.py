import plotly.graph_objects as go
import json


with open('tests_and_visualization/results_of_models/LK2.json', 'r', encoding='utf-8') as f:
    dct = json.load(f)

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=dct['fact']['x'],
    y=dct['fact']['y'],
    z=dct['fact']['z'],
    mode='lines',
    name='Фактическая траектория',

    line=dict(
        color='royalblue',
        width=8
    )
))


fig.add_trace(go.Scatter3d(
    x=dct['predict']['x'],
    y=dct['predict']['y'],
    z=dct['predict']['z'],
    mode='lines',
    name='Lucas-Kanade',
    
    line=dict(
        color='crimson',
        width=8
    )
))

for name, path in zip(['ORB', 'SIFT'], ['tests_and_visualization/results_of_models/ORB2.json', 'tests_and_visualization/results_of_models/SIFT2.json']):
    with open(path, 'r', encoding='utf-8') as f:
        dct = json.load(f)
    
    fig.add_trace(go.Scatter3d(
    x=dct['predict']['x'],
    y=dct['predict']['y'],
    z=dct['predict']['z'],
    mode='lines',
    name=name,
    
    line=dict(
        # color='crimson',
        width=8
    )
))

fig.update_layout(


    width=2400,
    height=1600,

    font=dict(
        size=18,
        color='black'
    ),


    legend=dict(
        font=dict(size=18)
    ),

    scene=dict(

        xaxis=dict(
            title='X [м]',
            title_font=dict(size=22),
            tickfont=dict(size=16),

            showgrid=True,
            gridcolor='black',
            gridwidth=3,

            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=4,

            showline=True,
            linecolor='black',
            linewidth=4,

            backgroundcolor='white'
        ),

        yaxis=dict(
            title='Y [м]',
            title_font=dict(size=22),
            tickfont=dict(size=16),

            showgrid=True,
            gridcolor='black',
            gridwidth=3,

            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=4,

            showline=True,
            linecolor='black',
            linewidth=4,

            backgroundcolor='white'
        ),

        zaxis=dict(
            title='Z [м]',
            title_font=dict(size=22),
            tickfont=dict(size=16),

            showgrid=True,
            gridcolor='black',
            gridwidth=3,

            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=4,

            showline=True,
            linecolor='black',
            linewidth=4,

            backgroundcolor='white'
        ),

        aspectmode='data'
    )
)

fig.show()