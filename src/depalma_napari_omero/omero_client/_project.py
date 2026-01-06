import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import skimage.io
from mousetumorpy import (
    NNUNET_MODELS,
    YOLO_MODELS,
    combine_images,
    generate_tracked_tumors,
    to_linkage_df,
)
from tqdm import tqdm

from depalma_napari_omero.omero_client._client import OmeroClient
from depalma_napari_omero.omero_client._compute import (
    _compute_tracking,
    _compute_roi,
    _compute_nnunet,
)
from depalma_napari_omero.omero_client._context import ImageContext, SpecimenContext
from depalma_napari_omero.omero_client.omero_config import OmeroConfig
from depalma_napari_omero.omero_client._tags_processor import TagsProcessor
from depalma_napari_omero.omero_client._scanner import ProjectScanner
import depalma_napari_omero.omero_client._utils as utils


class OmeroProjectManager:
    def __init__(
        self,
        omero_client: OmeroClient,
        project_id: int,
        project_name: str,
        launch_scan: bool,
    ):
        self.client = omero_client

        self.scanner = ProjectScanner(
            omero_client, project_id, project_name, launch_scan
        )

        self.id = project_id
        self.name = project_name

        # Creat the categorical tags if they do not exist
        self.image_tag_id = self.client.create_tag(self.id, "image")
        self.corrected_tag_id = self.client.create_tag(self.id, "corrected_pred")

    @property
    def lungs_models(self) -> List[str]:
        return list(YOLO_MODELS.keys())

    @property
    def tumor_models(self) -> List[str]:
        return list(NNUNET_MODELS.keys())

    def batch_roi(self, lungs_model: str, ask_confirm: bool = True) -> None:
        if not lungs_model in self.lungs_models:
            raise ValueError(
                f"⚠️ {lungs_model} is not an available model (available: {self.lungs_models})."
            )

        roi_missing_ctx: List[ImageContext] = self.scanner.view.roi_missing

        if len(roi_missing_ctx) == 0:
            print("No ROIs to compute.")
            return

        if ask_confirm:
            print("\n" + "-" * 60)
            print("The following image IDs will be used for ROI computation:")
            print(
                f"  → {', '.join(map(str, [ctx.image_id for ctx in roi_missing_ctx]))}"
            )
            print(f"\nThe resulting tumor masks will be uploaded to the OMERO project:")
            print(f"  → `{self.name}`")

            confirm = (
                input("\n✅ Press [Enter] to confirm, or type [n] to cancel: ")
                .strip()
                .lower()
            )
            print()

            if confirm == "n":
                return

        for _ in self._run_batch_roi(lungs_model, roi_missing_ctx):
            continue

    def _run_batch_roi(self, lungs_model: str, roi_missing_ctx: List[ImageContext]):
        with tqdm(total=len(roi_missing_ctx), desc="Computing ROIs") as pbar:
            for k, ctx in enumerate(roi_missing_ctx):
                print(
                    f"Computing {k+1} / {len(roi_missing_ctx)} ROIs. Image ID = {ctx.image_id}"
                )

                if ctx.image_name is None:
                    raise RuntimeError("Context should have an image name!")

                if ctx.image_id is None:
                    raise RuntimeError("Context should have an image ID!")

                if ctx.dataset_id is None:
                    raise RuntimeError("Context should have a dataset ID!")

                posted_image_name = f"{os.path.splitext(ctx.image_name)[0]}_roi.tif"

                _compute_roi(
                    model=lungs_model,
                    image_name=posted_image_name,
                    image_id=ctx.image_id,
                    dataset_id=ctx.dataset_id,
                    project_id=self.id,
                    omero_client=self.client,
                )

                pbar.update(1)
                yield k + 1

        self.scanner.update()

    def batch_nnunet(self, model: str, ask_confirm: bool = True) -> None:
        if not model in self.tumor_models:
            raise ValueError(
                f"⚠️ {model} is not an available model (available: {self.tumor_models})."
            )

        pred_missing_ctx: List[ImageContext] = self.scanner.view.pred_missing

        if len(pred_missing_ctx) == 0:
            print("No tumor moask to compute.")
            return

        if ask_confirm:
            print("\n" + "-" * 60)
            print("The following image IDs will be used for tumor mask computation:")
            print(
                f"  → {', '.join(map(str, [ctx.image_id for ctx in pred_missing_ctx]))}"
            )
            print(f"\nThe resulting tumor masks will be uploaded to the OMERO project:")
            print(f"  → `{self.name}`")

            confirm = (
                input("\n✅ Press [Enter] to confirm, or type [n] to cancel: ")
                .strip()
                .lower()
            )
            print()

            if confirm == "n":
                return

        for _ in self._run_batch_nnunet(model, pred_missing_ctx):
            continue

    def _run_batch_nnunet(self, model: str, pred_missing_ctx: List[ImageContext]):
        with tqdm(total=len(pred_missing_ctx), desc="Detecting tumors") as pbar:
            for k, ctx in enumerate(pred_missing_ctx):
                print(
                    f"Computing {k+1} / {len(pred_missing_ctx)} tumor predictions. Image ID = {ctx.image_id}"
                )

                if ctx.image_name is None:
                    raise RuntimeError("Context should have an image name!")

                if ctx.image_id is None:
                    raise RuntimeError("Context should have an image ID!")

                if ctx.dataset_id is None:
                    raise RuntimeError("Context should have a dataset ID!")

                posted_image_name = (
                    f"{os.path.splitext(ctx.image_name)[0]}_pred_nnunet_{model}.tif"
                )

                _compute_nnunet(
                    model=model,
                    image_name=posted_image_name,
                    image_id=ctx.image_id,
                    dataset_id=ctx.dataset_id,
                    project_id=self.id,
                    omero_client=self.client,
                )

                pbar.update(1)
                yield k + 1

        self.scanner.update()

    def batch_track(self):
        cases: List[str] = self.scanner.view.cases
        for _ in self._run_batch_tracking(cases):
            continue

    def _run_batch_tracking(self, cases: List[str]):
        k = 0
        with tqdm(total=len(cases), desc="Tracking tumors") as pbar:
            for specimen in cases:
                pbar.update(1)
                k += 1
                yield k

                ctx = self.get_specimen_context(specimen)

                # Skip if there is only one time point
                if ctx.n_labels < 2:
                    print(f"⚠️ Only one time point is available. Skipping tracking for this case: {specimen}.")
                    continue

                # Skip if there is already a table attachment
                if ctx.tracking_table_id is not None:
                    print(f"⚠️ Tracking table already exists (Table ID: {ctx.tracking_table_id}). Case: {specimen}. Skipping...")
                    continue
                
                # Skip if the tumor series IDs have NaN values
                if pd.isna(ctx.tumor_series).sum() > 0:
                    print(f"⚠️ Tumor series IDs has NaN values; tumors weren't computed in all scans? Skipping tracking for this case: {specimen}...")
                    continue

                _compute_tracking(
                    image_id=ctx.roi_series[0],  # Destination image is the first ROI
                    roi_timeseries_ids=ctx.roi_series,
                    tumor_timeseries_ids=ctx.tumor_series,
                    omero_client=self.client,
                )
        
    def handle_corrected_roi_uploaded(self, posted_image_id: int, image_id: int):
        img_tags = self.client.get_image_tags(image_id)
        
        exclude_tags = TagsProcessor.get_image_tags(img_tags)
        exclude_tags.append("roi")
        exclude_tags.append("raw_pred")

        self.client.copy_image_tags(
            src_image_id=image_id,
            dst_image_id=posted_image_id,
            exclude_tags=exclude_tags,
        )

        self.client.tag_image_with_tag(posted_image_id, tag_id=self.corrected_tag_id)

    def upload_from_parent_directory(self, parent_dir: Union[str, Path]):
        """Upload selecting the parent directory containing image directories to upload"""
        subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
        for k, image_dir in enumerate(subfolders):
            self.upload_from_directory(image_dir)
            yield k

    def upload_from_directory(self, image_dir: Union[str, Path]):
        # Read the 3D tiff (time and name are inferred from the file names)
        image_ctx = utils.load_ct_from_folder(image_dir)
        image_ctx.image_name = f"{image_ctx.specimen_tag}_{image_ctx.time_tag}"
        image_ctx.project_id = self.id

        self.scanner.upload_image(image_ctx, self.image_tag_id)

    def download_case(self, specimen: str, out_dir: Union[str, Path]):
        out_dir = Path(out_dir) / specimen
        if not out_dir.exists():
            os.makedirs(out_dir)
            print("Created the output folder: ", out_dir)

        ctx = self.get_specimen_context(specimen)

        # Download the image ROIs
        roi_out_file = out_dir / "rois_timeseries.tif"
        if not roi_out_file.exists():
            roi_images = []
            for roi_id in ctx.roi_series:
                print(f"Downloading roi (ID={roi_id})")
                image = self.client.download_image(roi_id)
                roi_images.append(image)
            rois_timeseries = combine_images(roi_images)
            skimage.io.imsave(str(roi_out_file), rois_timeseries)
        else:
            print(f"Already exists: {roi_out_file}")

        # Download the lungs
        # lungs_timeseries_list = []
        # for roi_id in roi_series:
        # print("Downloading the lungs annotation")
        # lungs = self.omero_client.download_binary_mask_from_image_rois(roi_id)
        # lungs_timeseries_list.append(lungs)
        # lungs_timeseries = combine_images(lungs_timeseries_list)
        # skimage.io.imsave(
        #     str(out_folder / "lungs_timeseries.tif"),
        #     lungs_timeseries,
        # )

        # Download the tumors
        tumor_out_file = out_dir / "tumors_untracked.tif"
        if not tumor_out_file.exists():
            tumor_images = []
            # Tumor series can have pandas NaNs in it... here, we ignore them
            valid_tumor_series_ids = [v for v in ctx.tumor_series if pd.notna(v)]
            if pd.isna(ctx.tumor_series).sum() > 0:
                print(f"⚠️ Tumor series IDs has NaN values; ignoring them (tumors weren't computed in all scans?).")
            for tumor_id in valid_tumor_series_ids:
                print(f"Downloading tumor mask (ID={tumor_id})")
                tumor = self.client.download_image(tumor_id)
                tumor_images.append(tumor)
            tumor_timeseries = combine_images(tumor_images)
            skimage.io.imsave(str(tumor_out_file), tumor_timeseries)
        else:
            print(f"Already exists: {tumor_out_file}")

        # Download the tracked tumors
        if ctx.tracking_table_id is not None:
            ts_out_file = out_dir / "tumors_tracked.tif"
            if not ts_out_file.exists():
                formatted_df = self.client.get_table(ctx.tracking_table_id)
                linkage_df = to_linkage_df(formatted_df)
                tumor_series_tracked = generate_tracked_tumors(tumor_timeseries, linkage_df)

                # Save the tracked tumors and CSV file
                skimage.io.imsave(str(ts_out_file), tumor_series_tracked)
                formatted_df.to_csv(str(out_dir / f"{specimen}_results.csv"))
            else:
                print(f"Already exists: {ts_out_file}")

    def download_all_cases(self, out_dir: Path):
        project_dir = out_dir / self.name
        if not project_dir.exists():
            os.makedirs(project_dir)
            print("Created the output folder: ", project_dir)

        for k, case in enumerate(self.scanner.view.cases):
            self.download_case(case, project_dir)
            yield k

    def get_specimen_context(self, specimen: str) -> SpecimenContext:
        roi_series, tumor_series = self.scanner.view.tumor_timeseries_ids(specimen)
        n_rois = len(roi_series)
        n_nan_labels = pd.isna(tumor_series).sum()
        n_labels = len(tumor_series) - n_nan_labels

        n_lungs = 0
        for roi_id in roi_series:  # Refers to the omero rois (it's confusing..)
            ome_roi_ids = self.client.get_image_rois(roi_id)
            if len(ome_roi_ids) == 1:
                n_lungs += 1  # TODO: correct logic?

        times = self.scanner.view.specimen_times(specimen)

        n_tracked = 0
        tracking_table_id = None
        if n_rois > 0:
            dst_image_id = roi_series[0]
            tracking_table_ids = self.client.get_image_table_ids(dst_image_id)
            if len(tracking_table_ids) == 1:
                n_tracked = n_labels
                tracking_table_id = tracking_table_ids[0]
            elif len(tracking_table_ids) > 1:
                print(f"Warning: Multiple tables are associated with this image: {roi_series[0]}. Which one is the tracking table? Skipping for now.")
    
        return SpecimenContext(
            name=specimen,
            times=times,
            n_rois=n_rois,
            n_labels=n_labels,
            n_lungs=n_lungs,
            n_tracked=n_tracked,
            roi_series=roi_series,
            tumor_series=tumor_series,
            tracking_table_id=tracking_table_id,
        )


class OmeroController(OmeroClient):
    def __init__(
        self,
        user: str,
        password: str,
        omero_cfg: Optional[OmeroConfig] = None,
    ):
        super().__init__(user=user, password=password, omero_cfg=omero_cfg)
        self.project_manager = None

    def set_project(
        self, project_id: int, project_name: str, launch_scan: bool
    ) -> OmeroProjectManager:
        self.project_manager = OmeroProjectManager(
            self, project_id, project_name, launch_scan
        )
        return self.project_manager
