from pathlib import Path
import param
import panel as pn
import holoviews as hv
import numpy as np
import io
from scipy.io import loadmat
from rs_fusion_datasets import hsi2rgb
from rs_fusion_datasets.util.fileio import load_one_key_mat
import einops
import rasterio
from holoviews import streams

pn.extension(sizing_mode="stretch_width", design="material")
pn.extension('filedropper', 'tabulator')

CACHE_PATH = Path("./tmp")

class HSIImagePreview(pn.viewable.Viewer):
    bytes_dict = param.Dict(default={}, allow_None=True)
    image_format = param.Selector(
        default="CHW",
        objects=["CHW", "HWC"],
    )

    def __panel__(self):
        if not self.bytes_dict:
            return pn.pane.Markdown("No HSI file")

        file_name = list(self.bytes_dict.keys())[-1]
        print(f"Processing file: {file_name}")

        suffix = file_name.split('.')[-1].lower() if '.' in file_name else ''
        match suffix:
            case "mat":
                hsi = load_one_key_mat(io.BytesIO(self.bytes_dict[file_name]))
            case "hdr" | '':
                stem = file_name.removesuffix('.hdr')
                raw_file = f"{stem}"
                hdr_file = f"{stem}.hdr"
                if raw_file not in self.bytes_dict or not hdr_file in self.bytes_dict:
                    print(f"Missing corresponding raw file or hdr file, uploaded {list(self.bytes_dict.keys())}")
                    return pn.pane.Markdown("Waiting for the raw file and hdr file to be uploaded together.")
                
                # save the hdr file and raw file to cache
                hdr_path = CACHE_PATH / hdr_file
                raw_path = CACHE_PATH / raw_file
                hdr_path.write_bytes(self.bytes_dict[hdr_file])
                raw_path.write_bytes(self.bytes_dict[raw_file])

                with rasterio.open(raw_path) as f:
                    hsi = f.read()
            case _:
                raise ValueError(f"Unsupported file type: {suffix}")
        if self.image_format == "HWC":
            hsi = einops.rearrange(hsi, 'H W C -> C H W')
        c,h,w = hsi.shape
        rgb = hsi2rgb(hsi, wavelength_range=(400, 1000), input_format="CHW", output_format="HWC", to_u8np=True)

        rgb_plot = hv.RGB(rgb).opts(width=w, height=h)
        return pn.pane.HoloViews(
            rgb_plot
        )
    

class App(pn.viewable.Viewer):
    file_dropper = pn.widgets.FileDropper(
        name="Drop files here",
        multiple=True,
        chunk_size=20_000_000,
    )
    image_format = pn.widgets.Select(
        name="Image Format",
        options=["CHW", "HWC"],
        value="CHW",
    )
    hsi_preview = param.ClassSelector(class_=HSIImagePreview)
    spectral_plots = pn.pane.HoloViews(
        hv.Curve([]).opts(width=600, height=300, title="Spectral Plot"),
    )

    def __panel__(self):
        return pn.Column(
            self.file_dropper,
            self.image_format,
            pn.Row(
                pn.bind(HSIImagePreview, bytes_dict=self.file_dropper.param.value, image_format=self.image_format.param.value),
                self.spectral_plots
            )
        )

if pn.state.served:
    CACHE_PATH.mkdir(exist_ok=True, parents=True)
    App().servable()