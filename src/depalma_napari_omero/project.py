import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from mousetumorpy import LungsPredictor, TumorPredictor
from depalma_napari_omero.configuration import OMERO_TAGS


class ProjectRepresentation:
    def __init__(self, server, project_id, project_name):
        self.id = project_id
        self.name = project_name
        self.server = server
        self.all_categories = ["image", "roi", "raw_pred", "corrected_pred"]
        self._parse()

    @property
    def cases(self):
        return list(self.df.get("specimen").unique())
    
    def _parse(self):
        n_datasets = self.server.get_n_datasets_in_project(self.id)

        dataset_ids = []
        dataset_names = []
        image_ids = []
        image_names = []
        specimens = []
        times = []
        time_tags = []
        image_classes = []
        previous_dataset_id = None

        with tqdm(total=n_datasets, desc="Scanning project") as pbar:
            for (
                dataset_id,
                dataset_name,
                image_id,
                image_name,
                specimen,
                time,
                time_tag,
                image_class,
            ) in self.server.project_data_generator(self.id):
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

        df = pd.DataFrame(
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
        df_other = df[df["class"] == "other"].copy()

        df = df[df["class"] != "other"]

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
                columns=pd.Index(["specimen", "time"] + self.all_categories, name="class"),
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
                columns=pd.Index(["specimen", "time"] + self.all_categories, name="class"),
                fill_value=0,
            )

        # Image but no roi
        roi_missing_anomalies = df_summary[
            (df_summary["image"] > 0) & (df_summary["roi"] == 0)
        ][["specimen", "time"]]
        merged = pd.merge(df, roi_missing_anomalies, on=["specimen", "time"], how="inner")
        roi_missing = merged[merged["class"] == "image"].sort_values(["specimen", "time"])[
            ["dataset_id", "image_id", "image_name", "specimen", "time", "class"]
        ]

        # Roi but no preds or corrections
        pred_missing_anomalies = df_summary[
            (df_summary["roi"] > 0)
            & (df_summary["raw_pred"] == 0)
            & (df_summary["corrected_pred"] == 0)
        ][["specimen", "time"]]
        merged = pd.merge(df, pred_missing_anomalies, on=["specimen", "time"], how="inner")
        pred_missing = merged[merged["class"] == "roi"].sort_values(["specimen", "time"])[
            ["dataset_id", "image_id", "image_name", "specimen", "time", "class"]
        ]

        # Preds but no corrections
        correction_missing_anomalies = df_summary[
            (df_summary["raw_pred"] > 0) & (df_summary["corrected_pred"] == 0)
        ][["specimen", "time"]]
        merged = pd.merge(
            df, correction_missing_anomalies, on=["specimen", "time"], how="inner"
        )

        self.df_summary = df_summary
        self.df = df
        self.merged = merged
        self.df_other = df_other
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

        correction_missing = self.merged[self.merged["class"] == "raw_pred"].sort_values(
            ["specimen", "time"]
        )[["dataset_id", "image_id", "specimen", "time", "class"]]
        n_correction_missing = len(correction_missing)

        n_images_other = len(self.df_other)
        n_specimens = self.df["specimen"].nunique()
        n_times = self.df["time"].nunique()
        valid_times = self.df["time"].unique()

        # Print a small report
        print(f"\n### Project summary - {self.name} (ID={self.id}) ###")
        print(f"Number of cases: {n_specimens}")
        print(f"Scan times: {n_times} ({valid_times})")
        
        # Warnings
        print("Warnings:")
        if n_correction_missing > 0:
            corrections_missing_ids = correction_missing["image_id"].tolist()
            print(f"- Corrected masks missing ({n_correction_missing}) for these image IDs: {corrections_missing_ids}")
        if n_images_other > 0:
            print(f"- {n_images_other} files could not be reliably tagged in {self.all_categories} and were added to the `Other files` list.")
        if n_removed_image_duplicate > 0:
            print(f"{n_removed_image_duplicate} specimen-time combinations have multiple associated `image` files matching and were skipped (IDs: {multiple_images_to_check})")
        if n_removed_image_missing > 0:
            print(f"{n_removed_image_missing} specimen-time combinations have no associated `image` files matching and were skipped (IDs: {image_missing_anomalies})")
        print()

    def batch_roi(self, model):
        self.server.connect()
        n_rois_to_compute = len(self.roi_missing)
        if n_rois_to_compute == 0:
            print("No ROIs to compute.")
            return

        roi_ids_to_compute = self.roi_missing["image_id"].tolist()
        print(f"Going to compute ROIs for these image IDs: {roi_ids_to_compute}")
        print(f"These ROIs will be uploaded to OMERO in project `{self.name}`.")
        confirm = input("\nPress any key to confirm or [n] to cancel.").strip().lower()
        if confirm == 'n':
            return

        predictor = LungsPredictor(model)
        with tqdm(total=n_rois_to_compute, desc="Computing ROIs") as pbar:
            for k, (_, row) in enumerate(
                self.roi_missing[["dataset_id", "image_id", "image_name"]].iterrows()
            ):
                image_id = row["image_id"]
                dataset_id = row["dataset_id"]
                image_name = row["image_name"]

                print(f"Computing {k+1} / {n_rois_to_compute} ROIs. Image ID = {image_id}")

                image_name_stem = os.path.splitext(image_name)[0]
                posted_image_name = f"{image_name_stem}_roi.tif"

                image = self.server.download_image(image_id)

                try:
                    predictor = LungsPredictor(model)  # There is only one model, for now
                    roi, lungs_roi = predictor.compute_3d_roi(
                        image
                    )  # Note - this is not doing any skip-level (set to 2 before)
                except:
                    print(
                        f"An error occured while computing the ROI in this image: ID={image_id}. Skipping..."
                    )
                    continue

                self.server.connect()
                posted_image_id = self.server.post_image_to_ds(
                    roi, dataset_id, posted_image_name
                )
                self.server.tag_image_with_tag(posted_image_id, tag_id=OMERO_TAGS["roi"])

                image_tags_list = self.server.find_image_tag(self.server.get_image_tags(image_id))
                self.server.copy_image_tags(
                    src_image_id=image_id,
                    dst_image_id=posted_image_id,
                    exclude_tags=image_tags_list,
                )

                pbar.update(1)

    def batch_nnunet(self, model):
        self.server.connect()
        n_preds_to_compute = len(self.pred_missing)
        if n_preds_to_compute == 0:
            print("Nothing to compute.")
            return

        pred_ids_to_compute = self.pred_missing["image_id"].tolist()
        print(f"Going to compute tumor masks for these image IDs: {pred_ids_to_compute}")
        print(f"These tumor masks will be uploaded to OMERO in project `{self.name}`.")
        confirm = input("\nPress any key to confirm or [n] to cancel.").strip().lower()
        if confirm == 'n':
            return
        
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

                image = self.server.download_image(image_id)

                try:
                    predictor = TumorPredictor(model)
                    image_pred = predictor.predict(image)
                except:
                    print(
                        f"An error occured while computing the NNUNET prediction in this image: ID={image_id}."
                    )
                    continue

                self.server.connect()
                posted_image_id = self.server.post_image_to_ds(
                    image_pred, dataset_id, posted_image_name
                )
                self.server.tag_image_with_tag(
                    posted_image_id, tag_id=OMERO_TAGS["pred_nnunet_v4"]
                )
                self.server.copy_image_tags(
                    src_image_id=image_id,
                    dst_image_id=posted_image_id,
                    exclude_tags=["roi"],
                )

                pbar.update(1)

    def image_timeseries_ids(self, specimen_name):
        """Returns the indeces of the labeled images in a timeseries. Priority to images with the #corrected tag, otherwise #raw_pred is used."""
        image_img_ids = self.df[(self.df["specimen"] == specimen_name) & (self.df["class"] == "image")][
            ["image_id", "time"]
        ]
        image_img_ids.sort_values(by="time", ascending=True, inplace=True)

        return image_img_ids["image_id"].tolist()