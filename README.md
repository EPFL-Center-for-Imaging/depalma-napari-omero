![EPFL Center for Imaging logo](https://imaging.epfl.ch/resources/logo-for-gitlab.svg)
![screenshot](./assets/screenshot.png)
# Image analysis of tumor nodules in mice CT scans

We provide a unified user interface in Napari to detect, track, visualize, annotate, and measure the size evolution of lung tumor nodules in mice CT scans. The datasets and experiment metadata are automatically downloaded and parsed from OMERO.

This project is part of a collaboration between the [EPFL Center for Imaging](https://imaging.epfl.ch/) and the [De Palma Lab](https://www.epfl.ch/labs/depalma-lab/).

## Installation

**As a standalone app**

Download and run the latest installer from the [Releases](https://github.com/EPFL-Center-for-Imaging/depalma-napari-omero/releases) page.

**In Python**

We recommend performing the installation in a clean Python environment. Install our package from PyPi:

```sh
pip install depalma-napari-omero
```

or from the repository:

```sh
pip install git+https://github.com/EPFL-Center-for-Imaging/depalma-napari-omero.git
```

or clone the repository and install with:

```sh
git clone git+https://github.com/EPFL-Center-for-Imaging/depalma-napari-omero.git
cd depalma-napari-omero
pip install -e .
```

## Usage

From the command-line, start Napari with the `depalma-napari-omero` plugin:

```
napari -w depalma-napari-omero
```

## License

This project is licensed under the [AGPL-3](LICENSE) license.

This project depends on the [ultralytics](https://github.com/ultralytics/ultralytics) package which is licensed under AGPL-3.

## Related projects

- [Mouse Tumor Net](https://github.com/EPFL-Center-for-Imaging/mousetumornet) | Detect tumor nodules in mice CT scans.
- [Mouse Lungs Seg](https://github.com/EPFL-Center-for-Imaging/mouselungseg) | Detect the lungs cavity in mice CT scans.
- [Mouse Tumor Track](https://github.com/EPFL-Center-for-Imaging/mousetumortrack) | Track tumor nodules in mice CT scans.
- [JupyterHub (De Palma)](https://gitlab.com/epfl-center-for-imaging/depalma-jupyterhub) | A tailored JupyterHub deployment.