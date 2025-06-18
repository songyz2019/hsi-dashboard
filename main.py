from itertools import product
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import dash_uploader as du
from rs_fusion_datasets.util.fileio import load_one_key_mat
from rs_fusion_datasets import hsi2rgb, fetch_trento
import numpy as np
from jaxtyping import Float, UInt8
from dash import callback_context
from scipy.io import savemat
from flask import send_file, Flask
import webbrowser

state_hsi :Float[np.ndarray, 'H W C'] = None
state_rgb :UInt8[np.ndarray, 'H W 3'] = None
state_selected_area = [0,0,0,0]
state_hold_spectral = False
state_line_color = "auto"
spec_fig = go.Figure()

server = Flask(__name__)
@server.route("/download-mat")
def download_mat():
    global state_hsi, state_selected_area
    if state_hsi is None:
        return "No data to download", 400

    x0, x1, y0, y1 = state_selected_area
    selected_data = state_hsi[y0:y1+1, x0:x1+1, :]
    selected_data = selected_data.transpose(2, 0, 1)
    savemat("./tmp/selected_area.mat", {"data": selected_data.astype('float32')}, do_compression=True)
    
    return send_file("./tmp/selected_area.mat", as_attachment=True, download_name="selected_area.mat")

def init():
    global state_hsi, state_rgb
    hsi, dsm, lbl, info = fetch_trento()
    state_hsi = hsi.transpose(1,2,0)
    state_rgb = hsi2rgb(state_hsi, wavelength=info["wavelength"], input_format='HWC', output_format='HWC', to_u8np=True)

def create_image_figure(rgb :UInt8[np.ndarray, 'H W 3']):
    fig = go.Figure()
    print(state_hold_spectral)
    fig.add_trace(
        go.Image(
            z=rgb,
            x0=0, y0=0,
            dx=1, dy=1,
            zmin=[0, 0, 0, 0], zmax=[255, 255, 255, 255],  # 假设 RGB 图像的值范围是 0-255
            name="HSI Image",
        )
    )

    fig.update_layout(
        dragmode="select",
        xaxis_autorange=True,
        yaxis_autorange="reversed",
        autosize=True,
    )

    return fig

app = dash.Dash(__name__, server=server)
du.configure_upload(app, "./uploads", use_upload_id=True)
app.layout = html.Div([
    du.Upload(
        id="upload-img",
        filetypes=["mat",],
    ),

    dcc.Graph(
        id="image-graph",
        figure=None,
        style={"width": "1000px"}
    ),
    
    dcc.Checklist(
        id="hold-spectral-checkbox",
        options=[{"label": "Hold Spectral", "value": "hold"}],
        value=[],
        style={"margin": "10px"}
    ),

    dcc.Dropdown(
        id="line-color",
        options=[
            {"label": "Auto", "value": "auto"},
            {"label": "Red", "value": "rgba(255,0,0, 0.5)"},
            {"label": "Green", "value": "rgba(0,255,0, 0.5)"},
            {"label": "Blue", "value": "rgba(0,0,255, 0.5)"},
            {"label": "Black", "value": "rgba(0,0,0, 0.5)"},
            {"label": "Orange", "value": "rgba(255,165,0, 0.5)"},
            {"label": "Purple", "value": "rgba(128,0,128, 0.5)"},
            {"label": "Yellow", "value": "rgba(255,255,0, 0.5)"},
            {"label": "Cyan", "value": "rgba(0,255,255, 0.5)"},
        ],
        value="auto",
        style={"margin": "10px"}
    ),

    dcc.Graph(
        id="spectral-graph",
        figure=None,
        style={"width": "1000px"}
    ),

    dcc.Download(
        id="download-mat",
        data=None,
    ),
    html.A(
        "Download Selected Area",
        href="/download-mat",
    )
])


@du.callback(
    Output("image-graph", "figure"),
    id="upload-img",
)
def update_image_graph(status: du.UploadStatus):
    if status is None:
        pass
    if not status.is_completed:
        return None
    global state_hsi, state_rgb
    state_hsi = load_one_key_mat(status.latest_file)
    state_rgb = hsi2rgb(state_hsi, wavelength_range=(400, 1000),input_format='HWC', output_format='HWC', to_u8np=True)
    return create_image_figure(state_rgb)


@app.callback(
    Output("spectral-graph", "figure"),
    [
        Input("image-graph", "clickData"),
        Input("image-graph", "selectedData")
    ]
)
def display_selection_info(clickData, selectedData):
    if selectedData is not None and "range" in selectedData and selectedData["range"]:
        xrange = selectedData["range"]["x"]
        yrange = selectedData["range"]["y"]
        x0, x1 = min(xrange), max(xrange)
        y0, y1 = min(yrange), max(yrange)
    elif clickData is not None and "points" in clickData and clickData["points"]:
        x,y = clickData["points"][0]["x"], clickData["points"][0]["y"]
        xrange = [x, x]
        yrange = [y, y]
    else:
        return None

    x0, x1 = min(xrange), max(xrange)
    y0, y1 = min(yrange), max(yrange)
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    if state_hsi is None:
        return None
    else:
        _, _, n_channel = state_hsi.shape

    global state_hold_spectral, state_rgb, spec_fig, state_line_color
    fig = spec_fig
    if not state_hold_spectral:
        fig.data = [] 
    x = np.arange(n_channel)
    step_x = max(1, (x1 - x0) // 15) # 控制 曲线的数量
    step_y = max(1, (y1 - y0) // 15)
    fig.update_yaxes(range=[float(state_hsi.min()), float(state_hsi.max())])
    for i,j in product(range(x0, x1+1, step_x), range(y0, y1+1, step_y)):
        fig.add_trace(
            go.Scatter(
            x=x,
            y=state_hsi[j, i, :],
            line=dict(
                color='rgba({},{},{}, 0.5)'.format(*state_rgb[j, i]) if state_line_color == "auto" else state_line_color,
            ),
            showlegend=False,
            )
        )
    fig.update_layout(
        title="Spectral Data (15*15 samples)" if step_x > 1 or step_y > 1 else "Spectral Data",
        xaxis_title="Wavelength Channel Index",
        yaxis_title="Reflectance",
        autosize=True,
    )
    global state_selected_area
    state_selected_area = [x0, x1, y0, y1]
    return fig

# checkbox
@app.callback(
    Input("hold-spectral-checkbox", "value")
)
def hold_spectral_checkbox(value):
    global state_hold_spectral, state_rgb, spec_fig
    if "hold" in value:
        state_hold_spectral = True
    else:
        state_hold_spectral = False

#line color
@app.callback(
    Output("line-color", "value"),
    Input("line-color", "value")
)
def update_line_color(line_color):
    print(f"Updating line color to: {line_color}")
    global state_line_color, spec_fig
    state_line_color = line_color
    return line_color

if __name__ == "__main__":
    # init()
    webbrowser.open("http://127.0.0.1:8050")
    app.run(debug=True)


