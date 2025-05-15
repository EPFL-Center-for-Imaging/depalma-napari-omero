import os

import numpy as np
from mousetumorpy import (  # Eventually this dependency should be removed from here
    NNUNET_MODELS, YOLO_MODELS, combine_images,
    generate_tracked_tumors, initialize_df, to_linkage_df)
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from napari.utils import DirectLabelColormap
from napari.utils.notifications import show_error, show_info, show_warning
from napari_toolkit.containers import setup_vcollapsiblegroupbox
from napari_toolkit.widgets import setup_colorpicker
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QLabel,
                            QLineEdit, QPushButton, QSpinBox, QTabWidget,
                            QVBoxLayout, QWidget)

from depalma_napari_omero.omero_client._project import OmeroController
from depalma_napari_omero.omero_widget._worker import WorkerManager


def get_labels_cmap(labels_data: np.ndarray, rgba: np.ndarray = np.array([0, 1, 0, 1])):
    color_dict = {}
    for idx in np.unique(labels_data):
        color_dict[idx] = rgba
    color_dict[0] = np.array([0, 0, 0, 0])
    color_dict[None] = np.array([0, 0, 0, 0])
    cmap = DirectLabelColormap(color_dict=color_dict)
    return cmap


def require_project(func):
    def wrapper(self, *args, **kwargs):
        if self.controller is None:
            show_warning("Select a projet first!")
            return
        elif self.project is None:
            show_warning("Select a projet first!")
            return
        return func(self, *args, **kwargs)

    return wrapper


class OMEROWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.controller = None

        ### Main layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        qwidget = QWidget(self)
        layout.addWidget(qwidget)

        ### Login
        login_layout = QGridLayout()
        login_layout.setContentsMargins(10, 10, 10, 10)
        login_layout.setAlignment(Qt.AlignTop)

        omero_groupbox, omero_container = setup_vcollapsiblegroupbox(
            None, "OMERO server", collapsed=True
        )

        grid_widget = QWidget()
        omero_layout = QGridLayout(grid_widget)
        omero_container.addWidget(grid_widget)

        # Omero server address
        omero_layout.addWidget(QLabel("URL", self), 0, 0)
        self.omero_server_ip = QLineEdit(self)
        self.omero_server_ip.setText("omero-server.epfl.ch")
        # self.omero_server_ip.setText("localhost")
        omero_layout.addWidget(self.omero_server_ip, 0, 1)

        # Omero group
        omero_layout.addWidget(QLabel("Group", self), 1, 0)
        self.omero_group = QLineEdit(self)
        self.omero_group.setText("imaging-updepalma")
        # self.omero_group.setText("system")
        omero_layout.addWidget(self.omero_group, 1, 1)

        # Omero port
        omero_layout.addWidget(QLabel("Port", self), 2, 0)
        self.omero_port = QSpinBox(self)
        self.omero_port.setMaximum(60_000)
        self.omero_port.setValue(4064)
        omero_layout.addWidget(self.omero_port, 2, 1)

        login_layout.addWidget(omero_groupbox, 0, 0, 1, 2)

        # # Remote compute server (soon)
        # remote_compute_groupbox, remote_compute_container = setup_vcollapsiblegroupbox(
        #     None, "Compute server", collapsed=True
        # )

        # grid_widget = QWidget()
        # compute_server_layout = QGridLayout(grid_widget)
        # remote_compute_container.addWidget(grid_widget)

        # compute_server_layout.addWidget(QLabel("URL", self), 0, 0)
        # self.comptue_server_ip = QLineEdit(self)
        # self.comptue_server_ip.setText("127.0.0.1")
        # compute_server_layout.addWidget(self.comptue_server_ip, 0, 1)

        # compute_server_layout.addWidget(QLabel("Use this server", self), 1, 0)
        # self.remote_compute_checkbox = QCheckBox()
        # self.remote_compute_checkbox.setChecked(False)
        # compute_server_layout.addWidget(self.remote_compute_checkbox, 1, 1)

        # login_layout.addWidget(remote_compute_groupbox, 1, 0, 1, 2)

        # Username
        login_layout.addWidget(QLabel("Username", self), 3, 0)
        self.username = QLineEdit(self)
        self.username.setText("imaging-robot")
        # self.username.setText("root")
        login_layout.addWidget(self.username, 3, 1)

        # Password
        login_layout.addWidget(QLabel("Password", self), 4, 0)
        self.password = QLineEdit(self)
        self.password.setEchoMode(QLineEdit.Password)
        # self.password.setText("omero")
        login_layout.addWidget(self.password, 4, 1)

        # Login
        login_btn = QPushButton("Login", self)
        login_btn.clicked.connect(self._login)
        login_layout.addWidget(login_btn, 5, 0, 1, 2)

        select_layout = QGridLayout()
        select_layout.setContentsMargins(10, 10, 10, 10)
        select_layout.setAlignment(Qt.AlignTop)

        # Experiment group
        experiment_group, experiment_container = setup_vcollapsiblegroupbox(
            None, "Experiment", collapsed=False
        )
        grid_widget = QWidget()
        experiment_layout = QGridLayout(grid_widget)
        experiment_container.addWidget(grid_widget)
        select_layout.addWidget(experiment_group, 0, 0)

        # Project (experiment)
        self.cb_project = QComboBox()
        self.cb_project.currentTextChanged.connect(self._on_project_change)
        experiment_layout.addWidget(self.cb_project, 0, 0, 1, 2)

        # Rescan
        self.btn_rescan = QPushButton("Rescan", self)
        self.btn_rescan.clicked.connect(self._reset_ui_and_update_project)
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
        self.btn_run_workflows.clicked.connect(self._run_all_workflows)
        experiment_layout.addWidget(self.btn_run_workflows, 3, 0, 1, 3)

        # Upload new scans
        self.btn_run_workflows = QPushButton("⬆️ Upload new scans", self)
        self.btn_run_workflows.clicked.connect(self._upload_new_scans)
        experiment_layout.addWidget(self.btn_run_workflows, 4, 0, 1, 3)

        # Download experiment
        self.btn_run_workflows = QPushButton("⬇️ Download project", self)
        self.btn_run_workflows.clicked.connect(self._download_experiment)
        experiment_layout.addWidget(self.btn_run_workflows, 5, 0, 1, 3)

        # Scan data group
        scan_data_group, scan_data_container = setup_vcollapsiblegroupbox(
            None, "Scan data", collapsed=False
        )
        grid_widget = QWidget()
        scan_data_layout = QGridLayout(grid_widget)
        scan_data_container.addWidget(grid_widget)
        select_layout.addWidget(scan_data_group, 1, 0)

        # Specimens
        self.cb_specimen = QComboBox()
        self.cb_specimen.currentTextChanged.connect(self._on_specimen_change)
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
        btn_download.clicked.connect(self._download_selected)
        scan_data_layout.addWidget(btn_download, 3, 0, 1, 2)

        # Upload layer input
        self.cb_upload = QComboBox()
        scan_data_layout.addWidget(QLabel("Corrected data", self), 4, 0)
        scan_data_layout.addWidget(self.cb_upload, 4, 1)

        # Upload button
        btn_upload_corrections = QPushButton("⏫ Upload", self)
        btn_upload_corrections.clicked.connect(self._upload_corrections)
        scan_data_layout.addWidget(btn_upload_corrections, 5, 0, 1, 2)

        # Tracking group
        timeseries_group, timeseries_container = setup_vcollapsiblegroupbox(
            None, "Time series", collapsed=False
        )
        grid_widget = QWidget()
        timeseries_layout = QGridLayout(grid_widget)
        timeseries_container.addWidget(grid_widget)
        select_layout.addWidget(timeseries_group, 2, 0)

        timeseries_layout.addWidget(QLabel("Selected case:", self), 0, 0)
        self.label_selected_case_value = QLabel("-", self)
        timeseries_layout.addWidget(self.label_selected_case_value, 0, 1)

        # Download ROIs timeseries
        timeseries_layout.addWidget(QLabel("Image series", self), 1, 0)
        self.btn_download_roi_series = QPushButton("⏬ (-)", self)
        self.btn_download_roi_series.clicked.connect(self._download_ts_rois)
        timeseries_layout.addWidget(self.btn_download_roi_series, 1, 1, 1, 2)

        # Download lungs timeseries
        timeseries_layout.addWidget(QLabel("Lungs series", self), 2, 0)
        self.btn_download_lungs_series = QPushButton("⏬ (-)", self)
        self.btn_download_lungs_series.clicked.connect(self._download_ts_lungs)
        timeseries_layout.addWidget(self.btn_download_lungs_series, 2, 1, 1, 2)

        # Download tumor series (untracked)
        timeseries_layout.addWidget(QLabel("Tumor series (untracked)", self), 3, 0)
        self.btn_download_untracked_tumors = QPushButton("⏬ (-)", self)
        self.btn_download_untracked_tumors.clicked.connect(
            self._download_untracked_tumors
        )
        timeseries_layout.addWidget(self.btn_download_untracked_tumors, 3, 1, 1, 2)

        # Download tracked tumor timeseries
        timeseries_layout.addWidget(QLabel("Tumor series (tracked)", self), 4, 0)
        self.btn_download_tracked_tumors = QPushButton("⏬ (-)", self)
        self.btn_download_tracked_tumors.clicked.connect(self._download_tracked_tumors)
        timeseries_layout.addWidget(self.btn_download_tracked_tumors, 4, 1, 1, 2)

        # Convert to Tracks
        self.cb_convert_to_tracks = QComboBox()
        timeseries_layout.addWidget(QLabel("Visualize Tracks", self), 5, 0)
        timeseries_layout.addWidget(self.cb_convert_to_tracks, 5, 1)
        self.btn_convert_to_tracks = QPushButton("Run", self)
        self.btn_convert_to_tracks.clicked.connect(self._convert_to_tracks)
        timeseries_layout.addWidget(self.btn_convert_to_tracks, 5, 2)

        # Convert to Binary
        self.cb_update_colormap = QComboBox()
        timeseries_layout.addWidget(QLabel("Change direct color", self), 6, 0)
        timeseries_layout.addWidget(self.cb_update_colormap, 6, 1)

        colorpicker_layout = QVBoxLayout()
        colorpicker = QWidget(self)
        colorpicker.setLayout(colorpicker_layout)
        self.colorpicker_widget = setup_colorpicker(
            layout=colorpicker_layout,
            initial_color=(0, 255, 0),
            function=self._update_direct_colormap,
        )
        timeseries_layout.addWidget(colorpicker, 6, 2)

        ### Generic upload tab
        generic_upload_layout = QGridLayout()
        generic_upload_layout.setContentsMargins(10, 10, 10, 10)
        generic_upload_layout.setAlignment(Qt.AlignTop)

        self.cb_dataset = QComboBox()
        self.cb_dataset.currentTextChanged.connect(self._on_dataset_change)
        generic_upload_layout.addWidget(QLabel("Dataset", self), 0, 0)
        generic_upload_layout.addWidget(self.cb_dataset, 0, 1)

        self.cb_download_generic = QComboBox()
        generic_upload_layout.addWidget(QLabel("Files", self), 1, 0)
        generic_upload_layout.addWidget(self.cb_download_generic, 1, 1)
        btn_download_generic = QPushButton("⏬ Download", self)
        btn_download_generic.clicked.connect(self._generic_download)
        generic_upload_layout.addWidget(btn_download_generic, 1, 2)

        self.cb_upload_generic = QComboBox()
        generic_upload_layout.addWidget(QLabel("Layer", self), 2, 0)
        generic_upload_layout.addWidget(self.cb_upload_generic, 2, 1)
        btn_upload_generic = QPushButton("⏫ Upload", self)
        btn_upload_generic.clicked.connect(self._generic_upload)
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

    @property
    def project(self):
        return self.controller.project_manager

    def _on_layer_change(self, e):
        self.cb_upload_generic.clear()
        for x in self.viewer.layers:
            if isinstance(x, Labels) | isinstance(x, Image):
                self.cb_upload_generic.addItem(x.name, x.data)

        self.cb_upload.clear()
        self.cb_convert_to_tracks.clear()
        self.cb_update_colormap.clear()

        for x in self.viewer.layers:
            if isinstance(x, Labels):
                if len(x.data.shape) == 3:
                    self.cb_upload.addItem(x.name, x.data)
                    self.cb_update_colormap.addItem(x.name, x.data)
                if len(x.data.shape) == 4:
                    self.cb_convert_to_tracks.addItem(x.name, x.data)
                    self.cb_update_colormap.addItem(x.name, x.data)

    def _login(self):
        host = self.omero_server_ip.text()
        group = self.omero_group.text()
        port = self.omero_port.value()
        user = self.username.text()
        password = self.password.text()

        # if self.remote_compute_checkbox.isChecked():
        #     compute_server_url = self.comptue_server_ip.text()
        # else:
        #     compute_server_url = None

        self.controller = OmeroController(
            host,
            group,
            port,
            user,
            password,
            # compute_server_url,
        )

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

    def _on_dataset_change(self, selected_dataset):
        """On dataset change, update the generic files dropdown."""
        if selected_dataset == "":
            return

        dataset_id = int(self.cb_dataset.currentData())
        titles, image_ids = self.project.cb_dataset_image_data(dataset_id)
        self.cb_download_generic.clear()
        for title, image_id in zip(titles, image_ids):
            self.cb_download_generic.addItem(title, image_id)

    def _on_project_change(self, selected_project):
        """On project change, create a new ProjectRepresentation object."""
        if selected_project in ["", "Select from list"]:
            return

        project_id = int(self.cb_project.currentData())

        self.controller.set_project(project_id, selected_project)

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
        self.worker_manager.add_active(worker, max_iter=self.project.n_datasets)

    @thread_worker
    def _update_project_worker(self):
        for step in self.project.launch_scan():
            yield step

        dataset_titles, dataset_data = self.project.dataset_data_and_titles()

        # Update the UI
        self.cb_specimen.addItems(self.project.cases)
        for title, data in zip(dataset_titles, dataset_data):
            self.cb_dataset.addItem(title, data)

    def _on_specimen_change(self, specimen):
        if specimen == "":
            return

        times = self.project.specimen_times(specimen)

        roi_timeseries_ids, labels_timeseries_ids = self.project.tumor_timeseries_ids(
            specimen
        )
        n_rois_timeseries = len(roi_timeseries_ids)
        n_nans_labels_timeseries = np.isnan(labels_timeseries_ids).any().sum()
        n_labels_timeseries = len(labels_timeseries_ids) - n_nans_labels_timeseries

        # Lungs series
        if self.check_lungs_available(roi_timeseries_ids):
            n_lungs_timeseries = n_rois_timeseries
        else:
            n_lungs_timeseries = 0

        # Tracked tumor IDs
        if len(roi_timeseries_ids):
            dst_image_id = roi_timeseries_ids[0]
            tracking_table_ids = self.controller.omero_client.get_image_table_ids(
                image_id=dst_image_id
            )
        else:
            tracking_table_ids = []

        # Update the UI
        self.cb_image.clear()
        self.cb_scan_time.clear()
        self.cb_scan_time.addItems(times)
        self.label_selected_case_value.setText(f"{specimen}")
        self.btn_download_roi_series.setText(f"⏬ {n_rois_timeseries} scans")
        self.btn_download_lungs_series.setText(f"⏬ {n_lungs_timeseries} scans")
        self.btn_download_untracked_tumors.setText(f"⏬ {n_labels_timeseries} scans")
        if len(tracking_table_ids) == 1:
            self.btn_download_tracked_tumors.setText(f"⏬ {n_labels_timeseries} scans")

    def _on_scan_time_change(self, selected_time):
        if selected_time == "":
            return

        specimen = self.cb_specimen.currentText()
        if specimen == "":
            return

        image_classes = self.project.specimen_image_classes(specimen, selected_time)

        self.cb_image.clear()
        self.cb_image.addItems(image_classes)

    @thread_worker
    def _download_worker(self, image_id, image_name, image_class):
        return (
            self.controller.omero_client.download_image(image_id),
            image_name,
            image_class,
        )

    @require_project
    def _generic_download(self, *args, **kwargs):
        if self.cb_download_generic.currentText() == "":
            return

        image_id = self.cb_download_generic.currentData()
        image_name = self.project.image_attribute_from_id(image_id, "image_name")
        image_class = self.project.image_attribute_from_id(image_id, "class")

        show_info(f"Downloading {image_id=} ({image_class})")
        worker = self._download_worker(image_id, image_name, image_class)
        worker.returned.connect(self._download_selected_returned)
        self.worker_manager.add_active(worker)

    @require_project
    def _download_selected(self, *args, **kwargs):
        image_class = self.cb_image.currentText()
        if image_class == "":
            return

        specimen = self.cb_specimen.currentText()
        if specimen == "":
            return

        time = self.cb_scan_time.currentText()
        if time == "":
            return

        image_id, image_name, dataset_id = self.project.find_in_df(
            specimen, time, image_class
        )

        show_info(f"Downloading {image_id=}")
        worker = self._download_worker(image_id, image_name, image_class)
        worker.returned.connect(self._download_selected_returned)
        self.worker_manager.add_active(worker)

    def _download_selected_returned(self, payload):
        """Callback from download thread returning."""
        image_data, image_name, image_class = payload
        if image_class in ["corrected_pred", "raw_pred"]:
            self.viewer.add_labels(
                image_data,
                name=image_name,
            )

        elif image_class in ["roi", "image"]:
            self.viewer.add_image(image_data, name=image_name)
        else:
            print(f"Unknown image class: {image_class}. Attempting to load an image.")
            self.viewer.add_image(image_data, name=image_name)

    @thread_worker
    def _upload_worker(self, posted_image_data, posted_image_name, dataset_id):
        posted_image_id = self.controller.omero_client.import_image_to_ds(
            posted_image_data,
            self.project.id,
            dataset_id,
            posted_image_name,
        )
        return posted_image_id

    @require_project
    def _upload_corrections(self, *args, **kwargs):
        """Handles uploading images to the OMERO server."""
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
        layer = self.viewer.layers[layer_name]
        layer_name = f"{os.path.splitext(layer_name)[0]}_corrected.tif"
        updated_data = layer.data.astype("uint8")  # No higher ints for OMERO

        image_id, image_name, dataset_id = self.project.find_in_df(
            specimen, time, image_class
        )

        worker = self._upload_worker(updated_data, layer_name, dataset_id)
        worker.returned.connect(self._upload_worker_returned)
        worker.returned.connect(
            lambda posted_image_id: self.project.handle_corrected_roi_uploaded(
                posted_image_id, image_id
            )
        )
        self.worker_manager.add_active(worker)

    def _upload_worker_returned(self, posted_image_id):
        self._reset_ui_and_update_project()
        show_info(f"Uploaded image {posted_image_id}.")

    @require_project
    def _generic_upload(self, *args, **kwargs):
        layer_name = self.cb_upload_generic.currentText()
        if layer_name == "":
            return

        selected_dataset_text = self.cb_dataset.currentText()
        if selected_dataset_text == "":
            show_warning("No dataset selected!")
            return

        layer = self.viewer.layers[layer_name]
        updated_data = layer.data
        if isinstance(layer, Labels):
            updated_data = updated_data.astype(np.uint8)

        dataset_id = int(self.cb_dataset.currentData())

        worker = self._upload_worker(updated_data, layer_name, dataset_id)
        worker.returned.connect(self._upload_worker_returned)
        self.worker_manager.add_active(worker)

    @require_project
    def _run_all_workflows(self, *args, **kwargs):
        if self.project.roi_missing is None:
            return

        n_rois_to_compute = len(self.project.roi_missing)

        lungs_model = self.cb_lungs_models.currentData()
        if lungs_model is None:
            show_warning("No lungs model available.")
            return

        tumor_model = self.cb_tumor_models.currentData()
        if tumor_model is None:
            show_warning("No tumor model available.")
            return

        n_tracks_to_compute = len(self.project.cases)

        worker = self._workflow_worker(
            lungs_model,
            n_rois_to_compute,
            tumor_model,
            n_tracks_to_compute,
        )

        worker.returned.connect(self._reset_ui_and_update_project)

        self.worker_manager.add_active(worker)

    @thread_worker
    def _workflow_worker(
        self, lungs_model, n_rois_to_compute, tumor_model, n_tracks_to_compute
    ):
        if n_rois_to_compute > 0:
            for _ in self.project._run_batch_roi(lungs_model, n_rois_to_compute):
                continue

        n_preds_to_compute = len(self.project.pred_missing)

        if n_preds_to_compute > 0:
            for _ in self.project._run_batch_nnunet(tumor_model, n_preds_to_compute):
                continue

        if n_tracks_to_compute > 0:
            for _ in self.project._run_batch_tracking(n_tracks_to_compute):
                continue

    @require_project
    def _reset_ui_and_update_project(self, *args, **kwargs):
        current_specimen_idx = self.cb_specimen.currentIndex()
        current_time_idx = self.cb_scan_time.currentIndex()
        current_dataset_idx = self.cb_dataset.currentIndex()

        worker = self._update_project_worker()

        # Select the dataset and time that was previously selected
        worker.returned.connect(
            lambda _: self.cb_specimen.setCurrentIndex(current_specimen_idx)
        )
        worker.returned.connect(
            lambda _: self.cb_scan_time.setCurrentIndex(current_time_idx)
        )
        worker.returned.connect(
            lambda _: self.cb_dataset.setCurrentIndex(current_dataset_idx)
        )

        self.worker_manager.add_active(worker, max_iter=self.project.n_datasets)

    @require_project
    def _download_ts_rois(self, *args, **kwargs):
        specimen_name = self.cb_specimen.currentText()
        if specimen_name == "":
            return

        roi_timeseries_ids, labels_timeseries_ids = self.project.tumor_timeseries_ids(
            specimen_name
        )

        if len(roi_timeseries_ids) == 0:
            show_warning("No data to download.")
            return

        worker = self._download_timeseries_worker(roi_timeseries_ids, specimen_name)
        worker.returned.connect(
            lambda payload: self.viewer.add_image(payload[0], name=f"{payload[1]}_rois")
        )
        self.worker_manager.add_active(worker, max_iter=len(roi_timeseries_ids))

    @thread_worker
    def _download_timeseries_worker(self, to_download_ids, specimen_name):
        images = []
        for k, img_id in enumerate(to_download_ids):
            print(f"Downloading image ID = {img_id}")
            images.append(self.controller.omero_client.download_image(img_id))
            yield k + 1

        return (combine_images(images), specimen_name)

    def check_lungs_available(self, roi_timeseries_ids):
        # There should be exactly one ROI attached for the lungs to be valid
        for roi_id in roi_timeseries_ids:
            all_roi_ids = self.controller.omero_client.get_image_rois(roi_id)
            if len(all_roi_ids) != 1:
                return False
        return True

    @require_project
    def _download_ts_lungs(self, *args, **kwargs):
        specimen_name = self.cb_specimen.currentText()
        if specimen_name == "":
            return

        roi_timeseries_ids, labels_timeseries_ids = self.project.tumor_timeseries_ids(
            specimen_name
        )

        if len(roi_timeseries_ids) == 0:
            show_warning("No data to download.")
            return

        if not self.check_lungs_available(roi_timeseries_ids):
            show_warning(f"No lungs data to download.")
            return

        worker = self._download_ts_lungs_worker(roi_timeseries_ids, specimen_name)
        worker.returned.connect(
            lambda payload: self.viewer.add_labels(
                payload[0],
                name=f"{payload[1]}_lungs",
            )
        )
        self.worker_manager.add_active(worker, max_iter=len(roi_timeseries_ids))

    @thread_worker
    def _download_ts_lungs_worker(self, to_download_ids, specimen_name):
        images = []
        for k, img_id in enumerate(to_download_ids):
            print(f"Downloading ROI from image ID = {img_id}")
            images.append(
                self.controller.omero_client.download_binary_mask_from_image_rois(
                    img_id
                )
            )
            yield k + 1

        return (combine_images(images), specimen_name)

    @require_project
    def _download_tracked_tumors(self, *args, **kwargs):
        specimen_name = self.cb_specimen.currentText()
        if specimen_name == "":
            return

        roi_timeseries_ids, labels_timeseries_ids = self.project.tumor_timeseries_ids(
            specimen_name
        )

        if len(labels_timeseries_ids) == 0:
            show_warning("No data to download.")
            return

        dst_image_id = roi_timeseries_ids[0]

        table_ids = self.controller.omero_client.get_image_table_ids(
            image_id=dst_image_id
        )
        if len(table_ids) != 1:
            show_warning("No data to download.")
            return

        table_ids = self.controller.omero_client.get_image_table_ids(
            image_id=dst_image_id
        )
        table_id = table_ids[0]  # Assuming there is only one table?

        worker = self._download_tracked_tumors_worker(
            labels_timeseries_ids, specimen_name, table_id
        )
        worker.returned.connect(
            lambda payload: self.viewer.add_labels(
                payload[0],
                name=f"{payload[1]}_tracked_tumors",
            )
        )
        self.worker_manager.add_active(worker, max_iter=len(roi_timeseries_ids))

    @thread_worker
    def _download_tracked_tumors_worker(
        self, labels_timeseries_ids, specimen_name, table_id
    ):
        tumor_timeseries = []
        for k, img_id in enumerate(labels_timeseries_ids):
            print(f"Downloading image ID = {img_id}")
            tumor_timeseries.append(self.controller.omero_client.download_image(img_id))
            yield k + 1

        tumor_timeseries = combine_images(tumor_timeseries)

        formatted_df = self.controller.omero_client.get_table(table_id)
        linkage_df = to_linkage_df(formatted_df)

        tumor_timeseries_tracked = generate_tracked_tumors(
            tumor_timeseries, linkage_df
        )

        return tumor_timeseries_tracked, specimen_name

    @require_project
    def _download_untracked_tumors(self, *args, **kwargs):
        specimen_name = self.cb_specimen.currentText()
        if specimen_name == "":
            return

        roi_timeseries_ids, labels_timeseries_ids = self.project.tumor_timeseries_ids(
            specimen_name
        )

        if len(labels_timeseries_ids) == 0:
            show_warning("No data to download.")
            return

        elif np.isnan(labels_timeseries_ids).any():
            show_error("Labels/Images mismatch!")
            print("IDs to download have NaNs! Run model predictions?")
            return

        worker = self._download_timeseries_worker(labels_timeseries_ids, specimen_name)
        worker.returned.connect(self._download_untracked_tumors_returned)
        self.worker_manager.add_active(worker, max_iter=len(labels_timeseries_ids))

    def _download_untracked_tumors_returned(self, payload):
        untracked_tumors_timeseries, specimen_name = payload
        untracked_tumors_timeseries = untracked_tumors_timeseries.astype(np.uint16)
        self.viewer.add_labels(
            untracked_tumors_timeseries,
            name=f"{specimen_name}_tumors",
        )

    def _convert_to_tracks(self, *args, **kwargs):
        labels_data = self.cb_convert_to_tracks.currentData()
        if labels_data is None:
            return

        df = initialize_df(labels_data, properties=["centroid", "label"])
        tracks_data = df[
            ["label", "frame_forward", "centroid-0", "centroid-1", "centroid-2"]
        ].values  # ID, T, Z, Y, X
        self.viewer.add_tracks(
            tracks_data,
            head_length=len(labels_data),
            tail_length=len(labels_data),
            name="Tracks",
        )

    def _update_direct_colormap(self, *args, **kwargs):
        labels_data = self.cb_update_colormap.currentData()
        if labels_data is None:
            return

        labels_layer = self.viewer.layers[self.cb_update_colormap.currentText()]
        rgba = np.array(list(self.colorpicker_widget.get_color()) + [255]) / 255
        cmap = get_labels_cmap(labels_data, rgba)
        labels_layer.colormap = cmap
        print("Updated colormap")

    @thread_worker
    def _upload_new_scans_worker(self, parent_dir):
        for k in self.project.upload_from_parent_directory(parent_dir):
            yield k + 1

    @require_project
    def _upload_new_scans(self, *args, **kwargs):
        parent_dir = QFileDialog.getExistingDirectory(
            self, caption="Mouse scans directory"
        )
        subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
        n_datasets_to_upload = len(subfolders)
        worker = self._upload_new_scans_worker(parent_dir)
        self.worker_manager.add_active(worker, max_iter=n_datasets_to_upload)

    @thread_worker
    def _download_experiment_worker(self, save_dir):
        for k in self.project.download_all_cases(save_dir):
            yield k + 1

        # Also save all tracking results in a single CSV
        self.project.save_merged_csv(save_dir)

    @require_project
    def _download_experiment(self, *args, **kwargs):
        save_dir = QFileDialog.getExistingDirectory(self, caption="Save experiment")
        print(f"{save_dir=}")
        worker = self._download_experiment_worker(save_dir)
        self.worker_manager.add_active(worker, max_iter=len(self.project.cases))
