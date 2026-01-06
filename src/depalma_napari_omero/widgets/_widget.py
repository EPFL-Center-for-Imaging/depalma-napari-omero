import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from mousetumorpy import (
    NNUNET_MODELS,
    YOLO_MODELS,
    combine_images,
    generate_tracked_tumors,
    to_linkage_df,
)
import napari
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_warning
from napari_toolkit.containers.collapsible_groupbox import QCollapsibleGroupBox
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from depalma_napari_omero.omero_client._context import ImageContext
from depalma_napari_omero.omero_client._project import (
    OmeroController,
    OmeroProjectManager,
)
from depalma_napari_omero.omero_client.omero_config import OmeroConfig
from depalma_napari_omero.widgets._worker import WorkerManager
from depalma_napari_omero.omero_client._scanner import ProjectScanner
from depalma_napari_omero.omero_client._view import ProjectDataView
import depalma_napari_omero.omero_client._utils as utils


class OMEROWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.controller = None

        default_omero_cfg = OmeroConfig()

        ### Main layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)  # type: ignore

        ### Login
        login_layout = QGridLayout()
        login_layout.setContentsMargins(10, 10, 10, 10)
        login_layout.setAlignment(Qt.AlignTop)  # type: ignore

        omero_groupbox = QCollapsibleGroupBox("OMERO server")  # type: ignore
        omero_groupbox.setChecked(False)
        omero_groupbox.toggled.connect(self.on_groupbox_toggled)
        omero_layout = QGridLayout(omero_groupbox)

        # Omero server address
        omero_layout.addWidget(QLabel("URL", self), 0, 0)
        self.omero_server_ip = QLineEdit(self)
        self.omero_server_ip.setText(default_omero_cfg.host)
        omero_layout.addWidget(self.omero_server_ip, 0, 1)

        # Omero group
        omero_layout.addWidget(QLabel("Group", self), 1, 0)
        self.omero_group = QLineEdit(self)
        self.omero_group.setText(default_omero_cfg.group)
        omero_layout.addWidget(self.omero_group, 1, 1)

        # Omero port
        omero_layout.addWidget(QLabel("Port", self), 2, 0)
        self.omero_port = QSpinBox(self)
        self.omero_port.setMaximum(60_000)
        self.omero_port.setValue(default_omero_cfg.port)
        omero_layout.addWidget(self.omero_port, 2, 1)

        login_layout.addWidget(omero_groupbox, 0, 0, 1, 2)

        # Username
        login_layout.addWidget(QLabel("Username", self), 3, 0)
        self.username = QLineEdit(self)
        self.username.setText(default_omero_cfg.default_user)
        login_layout.addWidget(self.username, 3, 1)

        # Password
        login_layout.addWidget(QLabel("Password", self), 4, 0)
        self.password = QLineEdit(self)
        self.password.setEchoMode(QLineEdit.Password)
        login_layout.addWidget(self.password, 4, 1)

        # Login
        login_btn = QPushButton("Login", self)
        login_btn.clicked.connect(self._login)
        login_layout.addWidget(login_btn, 5, 0, 1, 2)

        select_layout = QGridLayout()
        select_layout.setContentsMargins(10, 10, 10, 10)
        select_layout.setAlignment(Qt.AlignTop)  # type: ignore

        # Experiment group
        experiment_group = QCollapsibleGroupBox("Experiment")  # type: ignore
        experiment_group.setChecked(True)
        experiment_group.toggled.connect(self.on_groupbox_toggled)
        experiment_layout = QGridLayout(experiment_group)
        select_layout.addWidget(experiment_group, 0, 0)

        # Project (experiment)
        self.cb_project = QComboBox()
        self.cb_project.currentTextChanged.connect(self._on_project_change) # type: ignore
        experiment_layout.addWidget(self.cb_project, 0, 0, 1, 2)

        # Rescan
        self.btn_rescan = QPushButton("Rescan", self)
        self.btn_rescan.clicked.connect(self._reset_ui_and_update_project) # type: ignore
        experiment_layout.addWidget(self.btn_rescan, 0, 2)

        # Lungs model
        self.cb_lungs_models = QComboBox()
        for lungs_model_name in reversed(YOLO_MODELS.keys()):
            self.cb_lungs_models.addItem(lungs_model_name, lungs_model_name)
        experiment_layout.addWidget(QLabel("Lungs model", self), 1, 0)
        experiment_layout.addWidget(self.cb_lungs_models, 1, 1, 1, 2)

        # Tumor model
        self.cb_tumor_models = QComboBox()
        for tumor_model_name in reversed(NNUNET_MODELS.keys()):
            self.cb_tumor_models.addItem(tumor_model_name, tumor_model_name)
        experiment_layout.addWidget(QLabel("Tumor model", self), 2, 0)
        experiment_layout.addWidget(self.cb_tumor_models, 2, 1, 1, 2)

        # Run workflows
        self.btn_run_workflows = QPushButton("🔁 Run all workflows", self)
        self.btn_run_workflows.clicked.connect(self._run_all_workflows) # type: ignore
        experiment_layout.addWidget(self.btn_run_workflows, 3, 0, 1, 3)

        # Upload new scans
        self.btn_upload_scans = QPushButton("⬆️ Upload new scans", self)
        self.btn_upload_scans.clicked.connect(self._upload_new_scans) # type: ignore
        experiment_layout.addWidget(self.btn_upload_scans, 4, 0, 1, 3)

        # Download experiment
        self.btn_download_experiments = QPushButton("⬇️ Download project", self)
        self.btn_download_experiments.clicked.connect(self._download_experiment) # type: ignore
        experiment_layout.addWidget(self.btn_download_experiments, 5, 0, 1, 3)

        # Scan data group
        scan_data_group = QCollapsibleGroupBox("Scan data")  # type: ignore
        scan_data_group.setChecked(True)
        scan_data_group.toggled.connect(self.on_groupbox_toggled)
        scan_data_layout = QGridLayout(scan_data_group)
        select_layout.addWidget(scan_data_group, 1, 0)

        # Specimens
        self.cb_specimen = QComboBox()
        self.cb_specimen.currentTextChanged.connect(self._on_specimen_change) # type: ignore
        scan_data_layout.addWidget(QLabel("Case", self), 0, 0)
        scan_data_layout.addWidget(self.cb_specimen, 0, 1)

        # Scan time
        self.cb_scan_time = QComboBox()
        self.cb_scan_time.currentTextChanged.connect(self._on_scan_time_change)
        scan_data_layout.addWidget(QLabel("Scan time", self), 1, 0)
        scan_data_layout.addWidget(self.cb_scan_time, 1, 1)

        # Images (data class)
        self.cb_image = QComboBox()
        scan_data_layout.addWidget(QLabel("Data category", self), 2, 0)
        scan_data_layout.addWidget(self.cb_image, 2, 1)

        # Download button
        btn_download = QPushButton("⏬ Download", self)
        btn_download.clicked.connect(self._download_selected) # type: ignore
        scan_data_layout.addWidget(btn_download, 3, 0, 1, 2)

        # Upload layer input
        self.cb_upload = QComboBox()
        scan_data_layout.addWidget(QLabel("Corrected data", self), 4, 0)
        scan_data_layout.addWidget(self.cb_upload, 4, 1)

        # Upload button
        btn_upload_corrections = QPushButton("⏫ Upload", self)
        btn_upload_corrections.clicked.connect(self._upload_corrections) # type: ignore
        scan_data_layout.addWidget(btn_upload_corrections, 5, 0, 1, 2)

        # Tracking group
        self.timeseries_group = QCollapsibleGroupBox("Time series")  # type: ignore
        self.timeseries_group.setChecked(False)
        self.timeseries_group.toggled.connect(self.on_groupbox_toggled)
        timeseries_layout = QGridLayout(self.timeseries_group)
        select_layout.addWidget(self.timeseries_group, 2, 0)

        timeseries_layout.addWidget(QLabel("Selected case:", self), 0, 0)
        self.label_selected_case_value = QLabel("-", self)
        timeseries_layout.addWidget(self.label_selected_case_value, 0, 1)

        # Download ROIs timeseries
        timeseries_layout.addWidget(QLabel("Image series", self), 1, 0)
        self.btn_download_roi_series = QPushButton("⏬ (-)", self)
        self.btn_download_roi_series.clicked.connect(self._download_ts_rois) # type: ignore
        timeseries_layout.addWidget(self.btn_download_roi_series, 1, 1, 1, 2)

        # Download lungs timeseries
        timeseries_layout.addWidget(QLabel("Lungs series", self), 2, 0)
        self.btn_download_lungs_series = QPushButton("⏬ (-)", self)
        self.btn_download_lungs_series.clicked.connect(self._download_ts_lungs) # type: ignore
        timeseries_layout.addWidget(self.btn_download_lungs_series, 2, 1, 1, 2)

        # Download tumor series (untracked)
        timeseries_layout.addWidget(QLabel("Tumor series (untracked)", self), 3, 0)
        self.btn_download_untracked_tumors = QPushButton("⏬ (-)", self)
        self.btn_download_untracked_tumors.clicked.connect(
            self._download_untracked_tumors # type: ignore
        )
        timeseries_layout.addWidget(self.btn_download_untracked_tumors, 3, 1, 1, 2)

        # Download tracked tumor timeseries
        timeseries_layout.addWidget(QLabel("Tumor series (tracked)", self), 4, 0)
        self.btn_download_tracked_tumors = QPushButton("⏬ (-)", self)
        self.btn_download_tracked_tumors.clicked.connect(self._download_tracked_tumors) # type: ignore
        timeseries_layout.addWidget(self.btn_download_tracked_tumors, 4, 1, 1, 2)

        ### Generic upload tab
        generic_upload_layout = QGridLayout()
        generic_upload_layout.setContentsMargins(10, 10, 10, 10)
        generic_upload_layout.setAlignment(Qt.AlignTop)  # type: ignore

        self.cb_dataset = QComboBox()
        self.cb_dataset.currentTextChanged.connect(self._on_dataset_change)
        generic_upload_layout.addWidget(QLabel("Dataset", self), 0, 0)
        generic_upload_layout.addWidget(self.cb_dataset, 0, 1)

        self.cb_download_generic = QComboBox()
        generic_upload_layout.addWidget(QLabel("Files", self), 1, 0)
        generic_upload_layout.addWidget(self.cb_download_generic, 1, 1)
        btn_download_generic = QPushButton("⏬ Download", self)
        btn_download_generic.clicked.connect(self._generic_download) # type: ignore
        generic_upload_layout.addWidget(btn_download_generic, 1, 2)

        self.cb_upload_generic = QComboBox()
        generic_upload_layout.addWidget(QLabel("Layer", self), 2, 0)
        generic_upload_layout.addWidget(self.cb_upload_generic, 2, 1)
        btn_upload_generic = QPushButton("⏫ Upload", self)
        btn_upload_generic.clicked.connect(self._generic_upload) # type: ignore
        generic_upload_layout.addWidget(btn_upload_generic, 2, 2)

        ### Tabs
        tab1 = QWidget(self)
        tab1.setLayout(login_layout)
        tab2 = QWidget(self)
        tab2.setLayout(select_layout)
        tab3 = QWidget(self)
        tab3.setLayout(generic_upload_layout)
        self.tabs = QTabWidget()
        self.tabs.addTab(tab1, "Login")
        self.tabs.addTab(tab2, "Data selection")
        self.tabs.addTab(tab3, "Download / Upload")
        layout.addWidget(self.tabs)

        # Setup layer callbacks
        self.viewer.layers.events.inserted.connect(
            lambda e: e.value.events.name.connect(self._on_layer_change)
        )
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change(None)

        self.worker_manager = WorkerManager(grayout_ui_list=[tab2, tab3])

        cancel_btn = self.worker_manager.cancel_btn
        layout.addWidget(cancel_btn)
        pbar = self.worker_manager.pbar
        layout.addWidget(pbar)

    def on_groupbox_toggled(self, checked: bool):
        """Invalidate the (parent) dock widget's minimum size so that the layout can be shrunk fully on groupbox collapse."""
        self.parentWidget().setMinimumSize(0, 0)  # type: ignore

    @property
    def project(self) -> Optional[OmeroProjectManager]:
        if self.controller is None:
            show_warning("Login required!")
            return

        if self.controller.project_manager is None:
            show_warning("Project selection required!")
            return

        return self.controller.project_manager

    @property
    def scanner(self) -> ProjectScanner:
        if self.project is None:
            raise RuntimeError("Project selection required!")
        return self.project.scanner

    @property
    def view(self) -> ProjectDataView:
        return self.scanner.view

    def _on_layer_change(self, e):
        self.cb_upload_generic.clear()
        for x in self.viewer.layers:
            if isinstance(x, Labels) or isinstance(x, Image):
                self.cb_upload_generic.addItem(x.name, x.data)

        self.cb_upload.clear()
        for x in self.viewer.layers:
            if isinstance(x, Labels):
                if len(x.data.shape) == 3:
                    self.cb_upload.addItem(x.name, x.data)

    def _login(self):
        host = self.omero_server_ip.text()
        group = self.omero_group.text()
        port = self.omero_port.value()
        user = self.username.text()
        password = self.password.text()

        omero_cfg = OmeroConfig(port=port, host=host, group=group)
        self.controller = OmeroController(user, password, omero_cfg)

        connect_status = self.controller.connect()
        if connect_status:
            self.cb_project.clear()
            self.cb_project.addItem("Select from list", 0)

            for project_name, project_id in self.controller.projects.items():
                self.cb_project.addItem(project_name, project_id)

            self.tabs.setCurrentIndex(1)  # Select the data tab
        else:
            show_warning("Could not connect. Try again?")

        self.password.clear()

    def _on_dataset_change(self, selected_dataset: str):
        """On dataset change, update the generic files dropdown."""
        if selected_dataset == "":
            return

        dataset_data = self.cb_dataset.currentData()
        if dataset_data is None:
            show_warning("No dataset selected!")
            return
        dataset_id = int(dataset_data)

        titles, image_ids = self.view.cb_dataset_image_data(dataset_id)

        self.cb_download_generic.clear()
        for title, image_id in zip(titles, image_ids):
            self.cb_download_generic.addItem(title, image_id)

    def _on_project_change(self, selected_project: str):
        project_data = self.cb_project.currentData()
        if project_data is None:
            return
        project_id = int(project_data)
        if selected_project in ["", "Select from list"]:
            return

        project_id = int(self.cb_project.currentData())

        if self.controller is None:
            raise RuntimeError("Login required!")

        self.controller.set_project(project_id, selected_project, launch_scan=False)

        # Update the UI
        self.btn_download_roi_series.setText(f"⏬ (-)")
        self.btn_download_untracked_tumors.setText(f"⏬ (-)")
        self.btn_download_lungs_series.setText(f"⏬ (-)")
        self.btn_download_tracked_tumors.setText(f"⏬ (-)")
        self.label_selected_case_value.setText("-")
        self.cb_specimen.clear()
        self.cb_download_generic.clear()
        self.cb_scan_time.clear()
        self.cb_dataset.clear()

        worker = self._update_project_worker()
        self.worker_manager.add_active(worker, max_iter=self.scanner.n_datasets)

    @thread_worker
    def _update_project_worker(self):
        for step in self.scanner.launch_scan():
            yield step

        data, titles = self.view.dataset_data_and_titles()

        # Update the UI
        self.cb_specimen.addItems(self.view.cases)
        for t, d in zip(titles, data):
            self.cb_dataset.addItem(t, d)

    def _on_specimen_change(self, specimen: str):
        if self.project is None:
            return
        
        if specimen == "":
            return

        specimen_ctx = self.project.get_specimen_context(specimen) # type: ignore

        # Update the UI
        self.cb_scan_time.clear()
        self.cb_scan_time.addItems(specimen_ctx.times)
        self.label_selected_case_value.setText(f"{specimen}")
        self.btn_download_roi_series.setText(f"⏬ {specimen_ctx.n_rois} scans")
        self.btn_download_untracked_tumors.setText(f"⏬ {specimen_ctx.n_labels} scans")
        # Lungs series
        self.btn_download_lungs_series.setText(f"⏬ {specimen_ctx.n_lungs} scans")
        # Tracked tumors
        self.btn_download_tracked_tumors.setText(f"⏬ {specimen_ctx.n_tracked} scans")

    def _on_scan_time_change(self, selected_time: str):
        if selected_time == "":
            return

        specimen = self.cb_specimen.currentText()
        if specimen == "":
            return

        image_classes = self.view.specimen_image_classes(specimen, selected_time)

        self.cb_image.clear()
        self.cb_image.addItems(image_classes)

    @thread_worker
    def _download_worker(self, image_ctx: ImageContext):
        if self.project is None:
            return
        
        if image_ctx.image_id is None:
            raise RuntimeError("ID required to download image.")

        image_ctx.image = self.project.client.download_image(image_ctx.image_id)
        return image_ctx

    def _generic_download(self, *args, **kwargs):
        if self.project is None:
            return
        
        if self.cb_download_generic.currentText() == "":
            return

        image_id = self.cb_download_generic.currentData()
        image_name = self.view.image_attribute_from_id(image_id, "image_name")
        image_class = self.view.image_attribute_from_id(image_id, "class")

        image_ctx = ImageContext(
            image_class=image_class,
            image_id=image_id,
            image_name=image_name,
        )

        show_info(f"Downloading Image ID={image_ctx.image_id} ({image_ctx.image_class})")
        worker = self._download_worker(image_ctx) # type: ignore
        worker.returned.connect(self._download_selected_returned)
        self.worker_manager.add_active(worker)

    def _download_selected(self, *args, **kwargs):
        if self.project is None:
            return
        
        image_class = self.cb_image.currentText()
        if image_class == "":
            return

        specimen = self.cb_specimen.currentText()
        if specimen == "":
            return

        time = self.cb_scan_time.currentText()
        if time == "":
            return

        image_ctx = ImageContext(
            image_class=image_class,
            specimen_tag=specimen,
            time_idx=float(time),
        )

        image_ctx = self.view.complete(image_ctx)

        if image_ctx.image_id is None:
            raise RuntimeError("ID required to download image.")

        show_info(f"Downloading Image ID={image_ctx.image_id}")

        worker = self._download_worker(image_ctx) # type: ignore
        worker.returned.connect(self._download_selected_returned)
        self.worker_manager.add_active(worker)

    def _download_selected_returned(self, image_ctx: ImageContext) -> None:
        """Callback from download thread returning."""
        if image_ctx.image_class in ["corrected_pred", "raw_pred"]:
            self.viewer.add_labels(image_ctx.image, name=image_ctx.image_name)
        elif image_ctx.image_class in ["roi", "image"]:
            self.viewer.add_image(image_ctx.image, name=image_ctx.image_name)
        else:
            print(
                f"Unknown image class: {image_ctx.image_class}. Attempting to load an image."
            )
            self.viewer.add_image(image_ctx.image, name=image_ctx.image_name)

    @thread_worker
    def _upload_worker(self, image_ctx: ImageContext):
        if self.project is None:
            return
        if image_ctx.image is None:
            raise RuntimeError("Context needs an image array.")
        if image_ctx.project_id is None:
            raise RuntimeError("Context needs a project ID.")
        if image_ctx.dataset_id is None:
            raise RuntimeError("Context needs a dataset ID.")
        if image_ctx.image_name is None:
            raise RuntimeError("Context needs an image name.")

        posted_image_id = self.project.client.import_image_to_ds(
            image_ctx.image,
            image_ctx.project_id,
            image_ctx.dataset_id,
            image_ctx.image_name,
        )

        image_ctx.image_id = posted_image_id

        return image_ctx

    def _upload_corrections(self, *args, **kwargs):
        """Handles uploading images to the OMERO server."""
        if self.project is None:
            return
        
        layer_name = self.cb_upload.currentText()
        if layer_name == "":
            return

        image_class = self.cb_image.currentText()
        if image_class == "":
            return

        specimen = self.cb_specimen.currentText()
        if specimen == "":
            return

        time = self.cb_scan_time.currentText()
        if time == "":
            return

        # When uploading a corrected prediction, tag it like the original image, without the image tag itself.
        upload_name = f"{os.path.splitext(layer_name)[0]}_corrected.tif"
        # No higher ints for OMERO
        updated_data = self.viewer.layers[layer_name].data.astype("uint8")

        origin_ctx = ImageContext(
            image_class="raw_pred",
            specimen_tag=specimen,
            time_idx=float(time),
        )
        origin_ctx = self.view.complete(origin_ctx)
        
        original_image_id = origin_ctx.image_id
        
        upload_ctx = ImageContext(
            image_class="corrected_pred",
            project_id=self.project.id,
            dataset_id=origin_ctx.dataset_id,
            image_name=upload_name,
            specimen_tag=specimen,
            time_idx=float(time),
            image=updated_data,
            original_image_id=original_image_id,
        )
        
        worker = self._upload_worker(upload_ctx) # type: ignore
        worker.returned.connect(self._upload_correction_returned)
        self.worker_manager.add_active(worker)

    def _upload_correction_returned(self, image_ctx: ImageContext):
        if self.project is None:
            return
        
        if image_ctx.image_id is None:
            raise RuntimeError("Uploaded corrected image has no ID.")
        
        if image_ctx.original_image_id is None:
            raise RuntimeError("Context needs an original image ID.")

        self.project.handle_corrected_roi_uploaded(image_ctx.image_id, image_ctx.original_image_id)
        
        self._upload_worker_returned(image_ctx)
        show_info(f"Uploaded image {image_ctx.image_id}.")

    def _upload_worker_returned(self, image_ctx: ImageContext):
        self._reset_ui_and_update_project()
        show_info(f"Uploaded image {image_ctx.image_id}.")

    def _generic_upload(self, *args, **kwargs):
        if self.project is None:
            return
        
        layer_name = self.cb_upload_generic.currentText()
        if layer_name == "":
            return

        selected_dataset_text = self.cb_dataset.currentText()
        if selected_dataset_text == "":
            show_warning("No dataset selected!")
            return

        layer = self.viewer.layers[layer_name]
        if isinstance(layer, Labels):
            layer_data: Optional[np.ndarray] = layer.data # type: ignore
            if layer_data is None:
                show_warning(f"No data found in the layer ({layer_name}).")
                return
            updated_data = layer_data.astype(np.uint8)
        else:
            updated_data = layer.data

        dataset_data = self.cb_dataset.currentData()
        if dataset_data is None:
            show_warning("No dataset selected!")
            return
        dataset_id = int(dataset_data)

        image_ctx = ImageContext(
            image_class="unknown",
            image=updated_data,
            image_name=layer_name,
            dataset_id=dataset_id,
            project_id=self.project.id,
        )

        worker = self._upload_worker(image_ctx) # type: ignore
        worker.returned.connect(self._upload_worker_returned) # type: ignore
        self.worker_manager.add_active(worker)

    def _run_all_workflows(self, *args, **kwargs):
        if self.project is None:
            return

        roi_missing_ctx: List[ImageContext] = self.view.roi_missing

        lungs_model = self.cb_lungs_models.currentData()
        tumor_model = self.cb_tumor_models.currentData()

        worker = self._workflow_worker(lungs_model, roi_missing_ctx, tumor_model) # type: ignore

        worker.returned.connect(self._reset_ui_and_update_project)

        self.worker_manager.add_active(worker)

    @thread_worker
    def _workflow_worker(
        self,
        lungs_model: Optional[str],
        roi_missing_ctx: List[ImageContext],
        tumor_model: Optional[str],
    ):
        if self.project is None:
            return
        
        if len(roi_missing_ctx) > 0:
            if lungs_model is None:
                raise RuntimeError("Lungs model seletion required.")
            for _ in self.project._run_batch_roi(lungs_model, roi_missing_ctx):
                continue

        pred_missing_ctx: List[ImageContext] = self.view.pred_missing

        if len(pred_missing_ctx) > 0:
            if tumor_model is None:
                raise RuntimeError("Tumor model selection required.")
            for _ in self.project._run_batch_nnunet(tumor_model, pred_missing_ctx):
                continue

        if len(self.view.cases) > 0:
            for _ in self.project._run_batch_tracking(self.view.cases):
                continue

    def _reset_ui_and_update_project(self, *args, **kwargs):
        if self.project is None:
            return
        
        current_specimen_idx = self.cb_specimen.currentIndex()
        current_time_idx = self.cb_scan_time.currentIndex()
        current_dataset_idx = self.cb_dataset.currentIndex()

        worker = self._update_project_worker()
        worker.returned.connect(lambda _: self._reset_comboboxes(current_specimen_idx, current_time_idx, current_dataset_idx))
        self.worker_manager.add_active(worker, max_iter=self.scanner.n_datasets)

    def _reset_comboboxes(self, current_specimen_idx: int, current_time_idx: int, current_dataset_idx: int):
        self.cb_specimen.setCurrentIndex(current_specimen_idx)
        self._on_specimen_change(specimen=self.cb_specimen.currentText())
        self.cb_scan_time.setCurrentIndex(current_time_idx)
        self._on_scan_time_change(selected_time=self.cb_scan_time.currentText())
        self.cb_dataset.setCurrentIndex(current_dataset_idx)
        self._on_dataset_change(selected_dataset=self.cb_dataset.currentText())
    
    @thread_worker
    def _download_timeseries_worker(self, to_download_ids: List[int], specimen: str):
        if self.project is None:
            return
        
        images = []
        for k, img_id in enumerate(to_download_ids):
            print(f"Downloading image ID = {img_id}")
            images.append(self.project.client.download_image(img_id))
            yield k + 1

        return (combine_images(images), specimen)

    @thread_worker
    def _download_ts_lungs_worker(self, to_download_ids: List[int], specimen: str):
        if self.project is None:
            return
        
        images = []
        for k, img_id in enumerate(to_download_ids):
            print(f"Downloading ROI from image ID = {img_id}")
            images.append(
                self.project.client.download_binary_mask_from_image_rois(img_id)
            )
            yield k + 1

        return (combine_images(images), specimen)

    @thread_worker
    def _download_tracked_tumors_worker(
        self, to_download_ids: List[int], specimen: str, table_id: int
    ):
        if self.project is None:
            return
        
        tumor_timeseries = []
        for k, img_id in enumerate(to_download_ids):
            print(f"Downloading image ID = {img_id}")
            tumor_timeseries.append(self.project.client.download_image(img_id))
            yield k + 1

        tumor_timeseries = combine_images(tumor_timeseries)

        # Move this outside of the thread?
        formatted_df = self.project.client.get_table(table_id)
        linkage_df = to_linkage_df(formatted_df)

        tracked_tumors = generate_tracked_tumors(tumor_timeseries, linkage_df)

        return tracked_tumors, specimen
    
    def _download_ts_rois(self, *args, **kwargs):
        if self.project is None:
            return
        
        specimen = self.cb_specimen.currentText()
        if specimen == "":
            return

        specimen_ctx = self.project.get_specimen_context(specimen)
        if len(specimen_ctx.roi_series) == 0:
            show_warning("No data to download.")
            return

        worker = self._download_timeseries_worker(specimen_ctx.roi_series, specimen) # type: ignore
        worker.returned.connect(self._download_roi_series_returned) # type: ignore
        self.worker_manager.add_active(worker, max_iter=specimen_ctx.n_rois)
    
    def _download_roi_series_returned(self, payload: Tuple[np.ndarray, str]):
        roi_series, specimen = payload
        self.viewer.add_image(roi_series, name=f"{specimen}_rois")

    def _download_ts_lungs(self, *args, **kwargs):
        if self.project is None:
            return
        
        specimen = self.cb_specimen.currentText()
        if specimen == "":
            return

        specimen_ctx = self.project.get_specimen_context(specimen)
        if specimen_ctx.n_lungs == 0:
            show_warning(f"No data to download.")
            return

        worker = self._download_ts_lungs_worker(specimen_ctx.roi_series, specimen) # type: ignore
        worker.returned.connect(self._download_lungs_series_returned) # type: ignore
        self.worker_manager.add_active(worker, max_iter=specimen_ctx.n_lungs)
    
    def _download_lungs_series_returned(self, payload: Tuple[np.ndarray, str]):
        lungs_series, specimen = payload
        self.viewer.add_labels(lungs_series, name=f"{specimen}_lungs")

    def _download_untracked_tumors(self, *args, **kwargs):
        if self.project is None:
            return
        
        specimen = self.cb_specimen.currentText()
        if specimen == "":
            return

        specimen_ctx = self.project.get_specimen_context(specimen)
        if specimen_ctx.n_labels == 0:
            show_warning("No data to download.")
            return

        valid_tumor_series_ids = [v for v in specimen_ctx.tumor_series if pd.notna(v)]
        if pd.isna(specimen_ctx.tumor_series).sum() > 0:
            print(f"⚠️ Tumor series IDs has NaN values; ignoring them (tumors weren't computed in all scans?).")
        worker = self._download_timeseries_worker(
            valid_tumor_series_ids, specimen_ctx.name # type: ignore
        )
        worker.returned.connect(self._download_untracked_tumors_returned) # type: ignore
        self.worker_manager.add_active(worker, max_iter=specimen_ctx.n_labels)

    def _download_untracked_tumors_returned(self, payload: Tuple[np.ndarray, str]):
        data, specimen = payload
        self.viewer.add_labels(data.astype(np.uint16), name=f"{specimen}_tumors")

    def _download_tracked_tumors(self, *args, **kwargs):
        if self.project is None:
            return
        
        specimen = self.cb_specimen.currentText()
        if specimen == "":
            return

        specimen_ctx = self.project.get_specimen_context(specimen)
        if specimen_ctx.n_tracked == 0:
            show_warning("No data to download.")
            return

        table_id = specimen_ctx.tracking_table_id

        valid_tumor_series_ids = [v for v in specimen_ctx.tumor_series if pd.notna(v)]
        if pd.isna(specimen_ctx.tumor_series).sum() > 0:
            print(f"⚠️ Tumor series IDs has NaN values; ignoring them (tumors weren't computed in all scans?).")
        worker = self._download_tracked_tumors_worker(valid_tumor_series_ids, specimen, table_id) # type: ignore
        worker.returned.connect(self._download_tracked_tumors_returned) # type: ignore
        self.worker_manager.add_active(worker, max_iter=specimen_ctx.n_tracked)

    def _download_tracked_tumors_returned(self, payload: Tuple[np.ndarray, str]):
        data, specimen = payload
        self.viewer.add_labels(data.astype(np.uint16), name=f"{specimen}_tracked_tumors")

    @thread_worker
    def _upload_new_scans_worker(self, parent_dir: Union[Path, str]):
        if self.project is None:
            return
        
        for k in self.project.upload_from_parent_directory(parent_dir):
            yield k + 1

    def _upload_new_scans(self, *args, **kwargs):
        parent_dir = QFileDialog.getExistingDirectory(
            self, caption="Mouse scans directory"
        )
        if isinstance(parent_dir, str):
            if parent_dir != "":
                subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
                n_datasets_to_upload = len(subfolders)
                worker = self._upload_new_scans_worker(parent_dir) # type: ignore
                self.worker_manager.add_active(worker, max_iter=n_datasets_to_upload)

    @thread_worker
    def _download_experiment_worker(self, save_dir: str):
        if self.project is None:
            return
        
        save_path = Path(save_dir).resolve()
        
        for k in self.project.download_all_cases(save_path):
            yield k + 1

        # Also save all tracking results in a single CSV
        project_dir = save_path / self.project.name
        out_csv_path = project_dir / f"Project_{self.project.id}_tracking_results.csv"
        utils.save_merged_csv(project_dir, out_csv_path)
        show_info(f"Saved {out_csv_path}")

    def _download_experiment(self, *args, **kwargs):
        if self.project is None:
            return
        
        save_dir = QFileDialog.getExistingDirectory(self, caption="Save experiment")
        if isinstance(save_dir, str):
            if save_dir != "":
                print(f"{save_dir=}")
                worker = self._download_experiment_worker(save_dir) # type: ignore
                self.worker_manager.add_active(worker, max_iter=len(self.view.cases))
