import torch
import plotly.graph_objects as go
from dash import Dash, Patch, dcc, html, Input, Output, clientside_callback
from raytracer_2d import get_rect_subdivs, get_rects_verts_2d


rect = torch.tensor(
    [[32, 4], [8, 0], [0, 8]],
    dtype=torch.float32,
)
xtal_nsubs = torch.tensor([9, 9])

rect_subs = get_rect_subdivs(rect, xtal_nsubs)

plate_rects = torch.tensor(
    [
        [[29, 8], [2, 0], [0, 8]],
        [[29, -2], [2, 0], [0, 8]],
    ],
    dtype=torch.float32,
)
# define the field of view (FOV)
fov_npx = torch.tensor([4, 4])
fov_mmppx = torch.tensor([4, 4])
fov_dims = fov_npx * fov_mmppx
fov_corners = torch.tensor(
    [
        [-fov_dims[0], -fov_dims[1]],
        [fov_dims[0], -fov_dims[1]],
        [fov_dims[0], fov_dims[1]],
        [-fov_dims[0], fov_dims[1]],
    ]
).view(-1, 2)

fig = go.Figure()

rect_subs_verts = get_rects_verts_2d(rect_subs)
fig.update_layout(
    # hovermode="x unified",
    # xaxis=dict(showspikes=True),
    # yaxis=dict(showspikes=True),
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_scaleanchor="y",
    # yaxis_scaleanchor="x",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

fig.add_trace(
    go.Scatter(
        x=fov_corners[[0, 1, 2, 3, 0], 0],
        y=fov_corners[[0, 1, 2, 3, 0], 1],
        mode="lines",
        line=dict(color="black"),
        name="FOV",
        showlegend=False,
        hoverinfo="name",
    )
)
for i, verts in enumerate(rect_subs_verts):
    fig.add_trace(
        go.Scatter(
            x=verts[[0, 1, 2, 3, 0], 0],
            y=verts[[0, 1, 2, 3, 0], 1],
            mode="lines",
            line=dict(color="blue", width=1),
            name=f"{i}",
            # hoverinfo="name",
            fill="toself",
            fillcolor="rgba(0,0,255,0)",
            hoverinfo="none",
            showlegend=False,
        )
    )

plate_rects_verst = get_rects_verts_2d(plate_rects)
for i, verts in enumerate(plate_rects_verst):
    fig.add_trace(
        go.Scatter(
            x=verts[[0, 1, 2, 3, 0], 0],
            y=verts[[0, 1, 2, 3, 0], 1],
            mode="lines",
            line=dict(color="green", width=1),
            name=f"plate {i}",
            fill="toself",
            hoverinfo="name",
            showlegend=False,
        ),
    )

# Initialize a JupyterDash app
app = Dash(__name__)

# Define the app layout
app.layout = html.Div(
    [dcc.Graph(id="graph", figure=fig), html.Div(id="hover-output")]
)


# @app.callback(Output("hover-output", "children"), Input("graph", "hoverData"))
# def highlight_hover(hoverData):
#     if hoverData:
#         updated_fig = Patch()
#         polygon_id = hoverData["points"][0]["curveNumber"]
#         updated_fig.data[polygon_id].line.color = "red"
#         # hoverData.line.color = "red"
#         print(hoverData)
#         return f"Hovered polygon: {polygon_id}"
#     else:
#         return "Hover over a point"


clientside_callback(
    """
    function(hoverData) {
        if (hoverData) {
            var traceIndex = hoverData.points[0].curveNumber;
            // var originalColor = hoverData.points[0].data.line.color;
            // var hoverColor = 'yellow';
            // Plotly.restyle('graph', {'fillcolor': "rgba(0,0,255,1)"}, [traceIndex]);
            return traceIndex;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("hover-output", "children"),
    Input("graph", "hoverData"),
)

clientside_callback(
    """
    function(hoverData) {
        if (hoverData) {
            var traceIndex = hoverData.points[0].curveNumber;
            // var originalColor = hoverData.points[0].data.line.color;
            // var hoverColor = 'yellow';
            Plotly.restyle('graph', {'fillcolor': "rgba(0,0,255,1)"}, [traceIndex]);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("graph", "figure"),
    Input("graph", "hoverData"),
)


# Run the app
app.run_server()