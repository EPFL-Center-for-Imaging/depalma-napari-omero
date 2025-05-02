import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io
from imaging_server_kit.client import Client
from mousetumorpy import (
    NNUNET_MODELS,
    YOLO_MODELS,
    combine_images,
    generate_tracked_tumors,
    to_linkage_df,
)
from tqdm import tqdm

from depalma_napari_omero.omero_client import OmeroClient
from depalma_napari_omero.omero_client._compute import MouseTumorComputeServer


def find_specimen_tag(img_tags) -> str:
    """Finds a specimen name (e.g. 'C25065') among image tags based on a regular expression."""
    r = re.compile("C[0-9]+|Animal-[0-9]+")
    specimen_name_tag = list(sorted(filter(r.match, img_tags)))
    if len(specimen_name_tag) == 0:
        return None
    specimen_name_tag = specimen_name_tag[0]

    return specimen_name_tag


def find_image_tag(img_tags) -> list:
    r = re.compile("(I|i)mage(s?)")
    image_tag = list(filter(r.match, img_tags))
    if len(image_tag) == 0:
        return []

    return image_tag


def find_raw_pred_tag(img_tags) -> list:
    r = re.compile(".*pred.*")
    pred_tag = list(filter(r.match, img_tags))
    if len(pred_tag) == 0:
        return []

    return pred_tag


def find_scan_time_tag(img_tags) -> int:
    """Finds a time stamp (e.g. 'T2') among image tags based on a regular expression."""
    r = re.compile("(Tm?|SCAN|scan)[0-9]+")
    # r = re.compile("(Tm?|SCAN|scan)[0-9]+")  # TODO: Should we remove Tm1? = T?
    time_stamp_tag = list(sorted(filter(r.match, img_tags)))

    # There must be exactly one matching scan time tag
    if len(time_stamp_tag) != 1:
        if len(time_stamp_tag) > 1:
            print("Incoherent scan times: ", time_stamp_tag)
        return (np.nan, np.nan)
    time_stamp_tag = time_stamp_tag[0]

    t = re.findall(r"m?\d+", time_stamp_tag)[0]
    if t == "m1":  # Prescans - to be ignored during tracking?
        t = -1
    else:
        t = int(t)

    return (t, time_stamp_tag)


class OmeroProjectManager:
    def __init__(
        self, omero_client: OmeroClient, client, project_id: int, project_name: str
    ):
        self.all_categories = ["image", "roi", "raw_pred", "corrected_pred"]

        self.omero_client = omero_client
        self.client = client

        self.id = project_id
        self.name = project_name

        self.df_all = None
        self.df = None
        self.df_other = None
        self.df_summary = None
        self.df_merged = None
        self.roi_missing = None
        self.pred_missing = None

        # Creat the categorical tags if they do not exist
        self.image_tag_id = self.omero_client.create_tag_if_not_exists(self.id, "image")
        self.corrected_tag_id = self.omero_client.create_tag_if_not_exists(
            self.id, "corrected_pred"
        )

    @property
    def n_datasets(self):
        omero_project = self.omero_client.get_project(self.id)
        n_datasets = len(list(omero_project.listChildren()))
        return n_datasets

    @property
    def cases(self):
        if self.df is None:
            return
        return list(self.df.get("specimen").unique())

    def _project_data_generator(self):
        omero_project = self.omero_client.get_project(self.id)

        for dataset in omero_project.listChildren():
            dataset_id = dataset.getId()
            dataset_name = dataset.getName()
            dataset = self.omero_client.get_dataset(dataset_id)

            for image in dataset.listChildren():
                image_id = image.getId()
                image_name = image.getName()

                image_tags = self.omero_client.get_image_tags(image_id)

                specimen = find_specimen_tag(image_tags)

                time, time_tag = find_scan_time_tag(image_tags)

                if (specimen is None) | (time is np.nan):
                    image_class = "other"
                elif len(find_image_tag(image_tags)) >= 1:
                    image_class = "image"
                elif "roi" in image_tags:
                    image_class = "roi"
                elif ("corrected" in image_tags) | ("corrected_pred" in image_tags):
                    image_class = "corrected_pred"
                elif len(find_raw_pred_tag(image_tags)) >= 1:
                    image_class = "raw_pred"
                else:
                    image_class = "other"

                yield (
                    dataset_id,
                    dataset_name,
                    image_id,
                    image_name,
                    specimen,
                    time,
                    time_tag,
                    image_class,
                )

    def launch_scan(self):
        dataset_ids = []
        dataset_names = []
        image_ids = []
        image_names = []
        specimens = []
        times = []
        time_tags = []
        image_classes = []
        previous_dataset_id = None
        k = 0
        with tqdm(total=self.n_datasets, desc="Scanning project") as pbar:
            for (
                dataset_id,
                dataset_name,
                image_id,
                image_name,
                specimen,
                time,
                time_tag,
                image_class,
            ) in self._project_data_generator():
                dataset_ids.append(dataset_id)
                dataset_names.append(dataset_name)
                image_ids.append(image_id)
                image_names.append(image_name)
                specimens.append(specimen)
                times.append(time)
                time_tags.append(time_tag)
                image_classes.append(image_class)

                if (previous_dataset_id is None) or (previous_dataset_id != dataset_id):
                    previous_dataset_id = dataset_id
                    pbar.update(1)
                    k += 1
                    yield k

        df_all = pd.DataFrame(
            {
                "dataset_id": dataset_ids,
                "dataset_name": dataset_names,
                "image_id": image_ids,
                "image_name": image_names,
                "specimen": specimens,
                "time": np.array(times, dtype=float),  # .astype(int),
                "time_tag": time_tags,
                "class": image_classes,
            }
        )

        # Make a separate dataset out of the "other" class
        df_other = df_all[df_all["class"] == "other"].copy()
        df = df_all[df_all["class"] != "other"]

        df_summary = df.pivot_table(
            index=["specimen", "time"],
            columns="class",
            aggfunc="size",
            fill_value=0,
        ).reset_index()
        df_summary = df_summary.reindex(
            columns=pd.Index(["specimen", "time"] + self.all_categories, name="class"),
            fill_value=0,
        )

        # Remove rows with an image missing
        image_missing_anomalies = df_summary[df_summary["image"] == 0]
        if not image_missing_anomalies.empty:
            filt = df.set_index(["specimen", "time"]).index.isin(
                image_missing_anomalies.set_index(["specimen", "time"]).index
            )

            # Add the anomalies to the "df_other" dataset
            df_other = pd.concat([df_other, df[filt].copy()])

            # Remove the anomalies in df
            df = df[~filt].copy()
            df_summary = df.pivot_table(
                index=["specimen", "time"],
                columns="class",
                aggfunc="size",
                fill_value=0,
            ).reset_index()
            df_summary = df_summary.reindex(
                columns=pd.Index(
                    ["specimen", "time"] + self.all_categories, name="class"
                ),
                fill_value=0,
            )

        # Remove rows with multiple images
        multiple_image_anomalies = df_summary[df_summary["image"] > 1]

        if not multiple_image_anomalies.empty:
            filt = df.set_index(["specimen", "time"]).index.isin(
                multiple_image_anomalies.set_index(["specimen", "time"]).index
            )
            # Add the anomalies to the "df_other" dataset
            df_other = pd.concat([df_other, df[filt].copy()])

            # Remove the anomalies in df
            df = df[~filt].copy()
            df_summary = df.pivot_table(
                index=["specimen", "time"],
                columns="class",
                aggfunc="size",
                fill_value=0,
            ).reset_index()
            df_summary = df_summary.reindex(
                columns=pd.Index(
                    ["specimen", "time"] + self.all_categories, name="class"
                ),
                fill_value=0,
            )

        # Image but no roi
        roi_missing_anomalies = df_summary[
            (df_summary["image"] > 0) & (df_summary["roi"] == 0)
        ][["specimen", "time"]]
        df_merged = pd.merge(
            df, roi_missing_anomalies, on=["specimen", "time"], how="inner"
        )
        roi_missing = df_merged[df_merged["class"] == "image"].sort_values(
            ["specimen", "time"]
        )[["dataset_id", "image_id", "image_name", "specimen", "time", "class"]]

        # Roi but no preds or corrections
        pred_missing_anomalies = df_summary[
            (df_summary["roi"] > 0)
            & (df_summary["raw_pred"] == 0)
            & (df_summary["corrected_pred"] == 0)
        ][["specimen", "time"]]
        df_merged = pd.merge(
            df, pred_missing_anomalies, on=["specimen", "time"], how="inner"
        )
        pred_missing = df_merged[df_merged["class"] == "roi"].sort_values(
            ["specimen", "time"]
        )[["dataset_id", "image_id", "image_name", "specimen", "time", "class"]]

        # Preds but no corrections
        correction_missing_anomalies = df_summary[
            (df_summary["raw_pred"] > 0) & (df_summary["corrected_pred"] == 0)
        ][["specimen", "time"]]
        df_merged = pd.merge(
            df, correction_missing_anomalies, on=["specimen", "time"], how="inner"
        )

        self.df = df
        self.df_all = df_all

        self.df_summary = df_summary  # Used in print_summary()
        self.df_merged = df_merged  # Used in print_summary()
        self.df_other = df_other  # Used in print_summary()

        self.roi_missing = roi_missing
        self.pred_missing = pred_missing

    def print_summary(self):
        image_missing_anomalies = self.df_summary[self.df_summary["image"] == 0]
        n_removed_image_missing = len(image_missing_anomalies)

        multiple_image_anomalies = self.df_summary[self.df_summary["image"] > 1]

        filt = self.df.set_index(["specimen", "time"]).index.isin(
            multiple_image_anomalies.set_index(["specimen", "time"]).index
        )

        multiple_images_to_check = self.df[filt][self.df[filt]["class"] == "image"][
            ["specimen", "time", "time_tag", "class", "image_id"]
        ].sort_values(["specimen", "time"])

        n_removed_image_duplicate = len(multiple_image_anomalies)

        correction_missing = self.df_merged[
            self.df_merged["class"] == "raw_pred"
        ].sort_values(["specimen", "time"])[
            ["dataset_id", "image_id", "specimen", "time", "class"]
        ]
        n_correction_missing = len(correction_missing)

        n_images_other = len(self.df_other)
        n_specimens = self.df["specimen"].nunique()
        n_times = self.df["time"].nunique()
        valid_times = self.df["time"].unique()

        # Print a small report
        print("\n" + "=" * 60)
        print(f"📊 Project Summary: {self.name} (ID: {self.id})")
        print("=" * 60)
        print(f"🐭 Number of cases:      {n_specimens}")
        print(f"🕒 Scan times:           {n_times}")
        print("\n⚠️  Warnings:")

        if n_correction_missing > 0:
            corrections_missing_ids = ", ".join(
                map(str, correction_missing["image_id"].tolist())
            )
            print(f"  - {n_correction_missing} corrected masks missing for image IDs:")
            print(f"    {corrections_missing_ids}")
        if n_images_other > 0:
            print(
                f"  - {n_images_other} file(s) couldn't be reliably tagged in {self.all_categories}."
            )
            print(f"    Added to the `Other files` list.")
        if n_removed_image_duplicate > 0:
            ids = ", ".join(map(str, multiple_images_to_check))
            print(
                f"  - {n_removed_image_duplicate} specimen-time combinations had multiple matching `image` files."
            )
            print(f"    Skipped IDs: {ids}")
        if n_removed_image_missing > 0:
            ids = ", ".join(map(str, image_missing_anomalies))
            print(
                f"  - {n_removed_image_missing} specimen-time combinations had no matching `image` files."
            )
            print(f"    Skipped IDs: {ids}")

        if all(
            [
                n_correction_missing == 0,
                n_images_other == 0,
                n_removed_image_duplicate == 0,
                n_removed_image_missing == 0,
            ]
        ):
            print("  - No issues found 🎉")

        print("=" * 60 + "\n")

    @property
    def lungs_models(self):
        return list(YOLO_MODELS.keys())

    @property
    def tumor_models(self):
        return list(NNUNET_MODELS.keys())

    def batch_roi(self, model, ask_confirm=True):
        if not model in self.lungs_models:
            print(
                f"⚠️ {model} is not an available model (available: {self.lungs_models})."
            )
            return

        n_rois_to_compute = len(self.roi_missing)
        if n_rois_to_compute == 0:
            print("No ROIs to compute.")
            return

        roi_ids_to_compute = self.roi_missing["image_id"].tolist()

        if ask_confirm:
            print("\n" + "-" * 60)
            print("The following image IDs will be used for ROI computation:")
            print(f"  → {', '.join(map(str, roi_ids_to_compute))}")
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

        for _ in self._run_batch_roi(model, n_rois_to_compute):
            continue

    def _run_batch_roi(self, model, n_rois_to_compute):
        with tqdm(total=n_rois_to_compute, desc="Computing ROIs") as pbar:
            for k, (_, row) in enumerate(
                self.roi_missing[["dataset_id", "image_id", "image_name"]].iterrows()
            ):
                image_id = row["image_id"]
                dataset_id = row["dataset_id"]
                image_name = row["image_name"]

                print(
                    f"Computing {k+1} / {n_rois_to_compute} ROIs. Image ID = {image_id}"
                )

                image_name_stem = os.path.splitext(image_name)[0]
                posted_image_name = f"{image_name_stem}_roi.tif"

                self.client.run_algorithm(
                    workflow_step="roi",
                    lungs_model=model,
                    image_id=image_id,
                    dataset_id=dataset_id,
                    project_id=self.id,
                    posted_image_name=posted_image_name,
                )

                pbar.update(1)
                yield k + 1

        for _ in self.launch_scan():
            continue

    def batch_nnunet(self, model, ask_confirm=True):
        n_preds_to_compute = len(self.pred_missing)
        if n_preds_to_compute == 0:
            print("Nothing to compute.")
            return

        pred_ids_to_compute = self.pred_missing["image_id"].tolist()

        if ask_confirm:
            print("\n" + "-" * 60)
            print("The following image IDs will be used for tumor mask computation:")
            print(f"  → {', '.join(map(str, pred_ids_to_compute))}")
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

        for _ in self._run_batch_nnunet(model, n_preds_to_compute):
            continue

    def _run_batch_nnunet(self, model, n_preds_to_compute):
        with tqdm(total=n_preds_to_compute, desc="Detecting tumors") as pbar:
            for k, (_, row) in enumerate(
                self.pred_missing[["dataset_id", "image_id", "image_name"]].iterrows()
            ):
                image_id = row["image_id"]
                dataset_id = row["dataset_id"]
                image_name = row["image_name"]

                print(
                    f"Computing {k+1} / {n_preds_to_compute} tumor predictions. Image ID = {image_id}"
                )

                image_name_stem = os.path.splitext(image_name)[0]
                posted_image_name = f"{image_name_stem}_pred_nnunet_{model}.tif"

                self.client.run_algorithm(
                    workflow_step="pred",
                    tumor_model=model,
                    image_id=image_id,
                    dataset_id=dataset_id,
                    project_id=self.id,
                    posted_image_name=posted_image_name,
                )

                pbar.update(1)
                yield k + 1

        for _ in self.launch_scan():
            continue

    def batch_track(self):
        n_tracks_to_compute = len(self.cases)
        for _ in self._run_batch_tracking(n_tracks_to_compute):
            continue

    def _run_batch_tracking(self, n_tracks_to_compute):
        k = 0
        with tqdm(total=n_tracks_to_compute, desc="Tracking tumors") as pbar:
            for case in self.cases:
                pbar.update(1)
                k += 1
                yield k

                roi_timeseries_ids, tumor_timeseries_ids = self.tumor_timeseries_ids(
                    case
                )

                if len(roi_timeseries_ids) < 2:
                    continue

                dst_image_id = roi_timeseries_ids[0]

                # Skip if there is already a table attachment
                table_ids = self.omero_client.get_image_table_ids(image_id=dst_image_id)
                if len(table_ids) > 0:
                    continue

                self.client.run_algorithm(
                    workflow_step="track",
                    image_id=dst_image_id,
                    roi_timeseries_ids=roi_timeseries_ids,
                    tumor_timeseries_ids=tumor_timeseries_ids,
                )

    def images_timeseries_ids(self, specimen_name):
        """Returns the indeces of the images in a timeseries."""
        image_img_ids = self.df[
            (self.df["specimen"] == specimen_name) & (self.df["class"] == "image")
        ][["image_id", "time"]]
        image_img_ids.sort_values(by="time", ascending=True, inplace=True)

        return image_img_ids["image_id"].tolist(), image_img_ids["time"].tolist()

    def tumor_timeseries_ids(self, specimen_name):
        """Returns the indeces of the labeled images in a timeseries. Priority to images with the #corrected tag, otherwise #raw_pred is used."""

        def filter_group(group):
            if "corrected_pred" in group["class"].values:
                return group[group["class"] == "corrected_pred"].iloc[0]
            else:
                return group[group["class"] == "raw_pred"].iloc[0]

        roi_img_ids = self.df[
            (self.df["specimen"] == specimen_name) & (self.df["class"] == "roi")
        ][["image_id", "time"]]

        labels_img_ids = self.df[
            (self.df["specimen"] == specimen_name)
            & (self.df["class"].isin(["corrected_pred", "raw_pred"]))
        ][["image_id", "time", "class"]]

        labels_img_ids = (
            labels_img_ids.groupby("time").apply(filter_group).reset_index(drop=True)
        )

        labels_img_ids = pd.merge(
            roi_img_ids,
            labels_img_ids,
            on="time",
            how="left",
            suffixes=("_rois", "_labels"),
        )

        labels_img_ids.sort_values(by="time", ascending=True, inplace=True)

        return (
            labels_img_ids["image_id_rois"].tolist(),
            labels_img_ids["image_id_labels"].tolist(),
        )

    def specimen_times(self, specimen_name):
        sub_df = self.df[self.df["specimen"] == specimen_name]
        times = np.unique(sub_df["time"].tolist()).astype(str)
        return times

    def specimen_image_classes(self, specimen_name, time: str):
        time = int(
            float(time)
        )  # Handles '-1.0' which needs to be cast into a float first.

        sub_df = self.df[
            (self.df["specimen"] == specimen_name) & (self.df["time"] == time)
        ]
        image_classes = sub_df["class"].tolist()

        if ((sub_df["class"] == "roi").sum() > 1) | (
            (sub_df["class"] == "image").sum() > 1
        ):
            print("Duplicate images!")
            image_classes = []
        else:
            n_matches = len(image_classes)
            image_classes = np.unique(image_classes)
            if len(image_classes) != n_matches:
                print("Warning - Duplicate predictions found!")

        return image_classes

    def image_attribute_from_id(self, image_id, attribute):
        image_attribute = self.df_all[self.df_all["image_id"] == image_id][
            attribute
        ].tolist()[0]
        return image_attribute

    def find_in_df(self, specimen_name, time, image_class):
        time = int(
            float(time)
        )  # Handles '-1.0' which needs to be cast into a float first.

        sub_df = self.df[
            (self.df["specimen"] == specimen_name)
            & (self.df["time"] == time)
            & (self.df["class"] == image_class)
        ]

        image_id = sub_df["image_id"].tolist()[0]
        image_name = sub_df["image_name"].tolist()[0]
        dataset_id = sub_df["dataset_id"].tolist()[0]

        return image_id, image_name, dataset_id

    def cb_dataset_image_data(self, dataset_id):
        df_sorted = self.df_all[self.df_all["dataset_id"] == dataset_id].sort_values(
            by="image_id"
        )[["image_id", "image_name"]]
        df_sorted["title"] = df_sorted.apply(
            lambda row: f"{row['image_id']} - {row['image_name']}", axis=1
        )
        titles = df_sorted["title"].tolist()
        image_ids = df_sorted["image_id"].tolist()

        return titles, image_ids

    def dataset_data_and_titles(self):
        df_sorted = (
            self.df_all[["dataset_id", "dataset_name"]]
            .drop_duplicates()
            .sort_values(by="dataset_id")
        )
        df_sorted["title"] = df_sorted.apply(
            lambda row: f"{row['dataset_id']} - {row['dataset_name']}", axis=1
        )
        dataset_titles = df_sorted["title"].tolist()
        dataset_data = df_sorted["dataset_id"].tolist()

        return dataset_titles, dataset_data

    def handle_corrected_roi_uploaded(self, posted_image_id, image_id):
        img_tags = self.omero_client.get_image_tags(image_id)
        image_tags_list = find_image_tag(img_tags)
        image_tags_list.append("roi")
        image_tags_list.append("raw_pred")

        self.omero_client.copy_image_tags(
            src_image_id=image_id,
            dst_image_id=posted_image_id,
            exclude_tags=image_tags_list,
        )

        self.omero_client.tag_image_with_tag(
            posted_image_id, tag_id=self.corrected_tag_id
        )

    def upload_from_parent_directory(self, parent_dir: str):
        """Upload selecting the parent directory containing image directories to upload"""
        subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
        for k, image_dir in enumerate(subfolders):
            self.upload_from_directory(image_dir)
            yield k

    def upload_from_directory(self, image_dir: str):
        tiff_files = Path(image_dir).glob("*.tif")
        # Remove the file with "rec_spr" in the name which is an overview of the mouse
        tiff_image_files = sorted(
            [
                file
                for file in tiff_files
                if "rec_spr" not in Path(file).stem.split("~")[-1]
            ]
        )

        # Assuming that all files follow the naming convention, simply check the pattern in the first file
        exp_name, scan_time, specimen_name = (
            Path(tiff_image_files[0]).stem.split("~")[0].split("_")
        )

        # Read files into a 3D tiff
        image = np.array([skimage.io.imread(file) for file in tiff_image_files])

        # Upload the 3D tiff
        self.upload_new_scan(image, scan_time, specimen_name)

    def upload_new_scan(
        self, image: np.ndarray, scan_time: str, specimen_name: str
    ) -> int:
        # Check the naming conventions
        if find_specimen_tag([specimen_name]) is None:
            print(
                f"Specimen name {specimen_name} does not comply with naming convention (C*****)."
            )
            return

        if not isinstance(find_scan_time_tag([scan_time])[0], int):
            print(
                f"Scan time {scan_time} does not comply with naming convention (SCAN* or T*)."
            )
            return

        # Check if there is a dataset with the specimen name already - if so, use it
        if specimen_name in self.cases:
            dataset_ids = (
                self.df[self.df["specimen"] == specimen_name].get("dataset_id").unique()
            )
            if len(dataset_ids) > 1:
                print(
                    f"Multiple datasets found for this specimen name ({specimen_name})."
                )
                return

            dataset_id = dataset_ids[0]  # It should be the first (and unique) item
        else:
            # Otherwise, create a new dataset named like the specimen
            dataset_id = self.omero_client.post_dataset(
                project_id=self.id,
                dataset_name=specimen_name,
            )

        dataset_id = int(dataset_id)  # Convert numpy int to python int

        image_title = f"{specimen_name}_{scan_time}"

        # Post the image in the dataset
        posted_image_id = self.omero_client.import_image_to_ds(
            image=image,
            project_id=self.id,
            dataset_id=dataset_id,
            image_title=image_title,
        )

        # Tag the image appropriately
        scan_time_tag_id = self.omero_client.create_tag_if_not_exists(
            self.id, scan_time
        )
        specimen_tag_id = self.omero_client.create_tag_if_not_exists(
            self.id, specimen_name
        )
        project_tag_id = self.omero_client.create_tag_if_not_exists(self.id, self.name)

        self.omero_client.tag_image_with_tag(posted_image_id, tag_id=self.image_tag_id)
        self.omero_client.tag_image_with_tag(posted_image_id, tag_id=scan_time_tag_id)
        self.omero_client.tag_image_with_tag(posted_image_id, tag_id=specimen_tag_id)
        self.omero_client.tag_image_with_tag(posted_image_id, tag_id=project_tag_id)

        for _ in self.launch_scan():
            continue

        return dataset_id

    def download_case(self, selected_case, out_folder):
        out_folder = Path(out_folder) / selected_case
        if not out_folder.exists():
            os.makedirs(out_folder)
            print("Created the output folder: ", out_folder)

        roi_timeseries_ids, tumor_timeseries_ids = self.tumor_timeseries_ids(
            selected_case
        )

        rois_timeseries_list = []
        lungs_timeseries_list = []
        for roi_id in roi_timeseries_ids:
            print(f"Downloading roi (ID={roi_id})")
            image = self.omero_client.download_image(roi_id)
            rois_timeseries_list.append(image)

            # print("Downloading the lungs annotation")
            # lungs = self.omero_client.download_binary_mask_from_image_rois(roi_id)
            # lungs_timeseries_list.append(lungs)

        tumor_timeseries_list = []
        for tumor_id in tumor_timeseries_ids:
            print(f"Downloading tumor mask (ID={tumor_id})")
            tumor = self.omero_client.download_image(tumor_id)
            tumor_timeseries_list.append(tumor)

        rois_timeseries = combine_images(rois_timeseries_list)
        # lungs_timeseries = combine_images(lungs_timeseries_list)
        tumor_timeseries = combine_images(tumor_timeseries_list)

        skimage.io.imsave(
            str(out_folder / "rois_timeseries.tif"),
            rois_timeseries,
        )

        # skimage.io.imsave(
        #     str(out_folder / "lungs_timeseries.tif"),
        #     lungs_timeseries,
        # )

        skimage.io.imsave(
            str(out_folder / "tumors_untracked.tif"),
            tumor_timeseries,
        )

        dst_image_id = roi_timeseries_ids[0]
        table_ids = self.omero_client.get_image_table_ids(image_id=dst_image_id)

        if len(table_ids) == 1:
            table_id = table_ids[0]  # Assuming there is only one table?
            formatted_df = self.omero_client.get_table(table_id)
            linkage_df = to_linkage_df(formatted_df)

            tumor_timeseries_tracked = generate_tracked_tumors(
                tumor_timeseries, linkage_df
            )

            skimage.io.imsave(
                str(out_folder / "tumors_tracked.tif"),
                tumor_timeseries_tracked,
            )

            formatted_df.to_csv(str(out_folder / f"{selected_case}_results.csv"))
        else:
            print(
                f"{len(table_ids)} tables found attached to {dst_image_id=} (expected 1)."
            )

    def download_all_cases(self, save_dir):
        for k, case in enumerate(self.cases):
            self.download_case(case, save_dir)
            yield k

    def save_merged_csv(self, save_dir):
        all_dfs = []
        csv_files_in_save_dir = list(Path(save_dir).rglob("*.csv"))
        for csv_file in csv_files_in_save_dir:
            specimen_name = csv_file.stem.split("_")[0]

            if find_specimen_tag([specimen_name]) is None:
                print(
                    f"Specimen name {specimen_name} does not comply with naming convention (C*****)."
                )
                continue

            df_specimen = pd.read_csv(csv_file)
            df_specimen["Mouse_ID"] = specimen_name
            all_dfs.append(df_specimen)

        merged_df = pd.concat(all_dfs, ignore_index=True)
        if "Unnamed: 0" in merged_df.columns:
            merged_df = merged_df.drop(columns=["Unnamed: 0"])
        columns = ["Mouse_ID"] + [col for col in merged_df.columns if col != "Mouse_ID"]
        merged_df = merged_df[columns]
        out_csv_path = Path(save_dir) / f"Project_{self.id}_tracking_results.csv"
        merged_df.to_csv(out_csv_path)
        print(f"Saved {out_csv_path}")


class OmeroController:
    def __init__(
        self,
        host: str = None,
        group: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        compute_server_url: str = None,
    ):
        self.omero_client = OmeroClient(host, group, port)
        self.omero_client.login(user, password)

        if compute_server_url is None:
            self.client = MouseTumorComputeServer(host, group, port)
            self.client.login(user, password)
        else:
            self.client = Client(compute_server_url)

        self.project_manager = None

    def connect(self):
        return self.omero_client.connect()

    def quit(self):
        self.omero_client.quit()

    @property
    def projects(self):
        return self.omero_client.projects

    def set_project(
        self, project_id: int, project_name: str, launch_scan: bool = False
    ):
        self.project_manager = OmeroProjectManager(
            omero_client=self.omero_client,
            client=self.client,
            project_id=project_id,
            project_name=project_name,
        )
        if launch_scan:
            for _ in self.project_manager.launch_scan():
                continue
