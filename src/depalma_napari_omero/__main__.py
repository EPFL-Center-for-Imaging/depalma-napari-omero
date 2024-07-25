import napari
from depalma_napari_omero.omero_widget import OMEROWidget
from depalma_napari_omero import __version__

if __name__ == "__main__":
    viewer = napari.Viewer(title=f"De Palma Lab ({__version__})")
    viewer.window.add_dock_widget(OMEROWidget(viewer), name="De Palma Lab")
    napari.run()
