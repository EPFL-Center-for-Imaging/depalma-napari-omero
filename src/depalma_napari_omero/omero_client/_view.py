from typing import Any, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from depalma_napari_omero.omero_client._context import ImageContext


@dataclass
class ReportData:
    n_specimens: int
    n_times: int
    other_files: pd.DataFrame
    all_categories: List[str]
    corr_missing_ids: List[int]
    anomalous_multi_image: List[str]
    anomalous_image_missing: List[str]


def _find_roi_missing(df: pd.DataFrame, df_summary: pd.DataFrame) -> List[ImageContext]:
    roi_missing_anomalies = df_summary[
        (df_summary["image"] > 0) & (df_summary["roi"] == 0)
    ][["specimen", "time"]]

    df_merged = pd.merge(
        df, roi_missing_anomalies, on=["specimen", "time"], how="inner"
    )
    roi_missing = df_merged[df_merged["class"] == "image"].sort_values(
        ["specimen", "time"]
    )[["dataset_id", "image_id", "image_name", "specimen", "time", "class"]]

    rois_ctx = []
    for _, row in roi_missing.iterrows():
        roi_ctx = ImageContext(
            image_class="image",
            dataset_id=row["dataset_id"],
            image_id=row["image_id"],
            image_name=row["image_name"],
            specimen_tag=row["specimen"],
            time_tag=row["time"],
        )
        rois_ctx.append(roi_ctx)

    return rois_ctx


def _find_pred_missing(
    df: pd.DataFrame, df_summary: pd.DataFrame
) -> List[ImageContext]:
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

    pred_ctx = []
    for _, row in pred_missing.iterrows():
        roi_ctx = ImageContext(
            image_class="roi",
            dataset_id=row["dataset_id"],
            image_id=row["image_id"],
            image_name=row["image_name"],
            specimen_tag=row["specimen"],
            time_tag=row["time"],
        )
        pred_ctx.append(roi_ctx)

    return pred_ctx


def _find_corr_missing(df: pd.DataFrame, df_summary: pd.DataFrame):
    correction_missing_anomalies = df_summary[
        (df_summary["raw_pred"] > 0) & (df_summary["corrected_pred"] == 0)
    ][["specimen", "time"]]
    df_merged = pd.merge(
        df, correction_missing_anomalies, on=["specimen", "time"], how="inner"
    )
    corr_missing = df_merged[df_merged["class"] == "raw_pred"].sort_values(
        ["specimen", "time"]
    )[["dataset_id", "image_id", "image_name", "specimen", "time", "class"]]

    corr_ctx = []
    for _, row in corr_missing.iterrows():
        roi_ctx = ImageContext(
            image_class="raw_pred",
            dataset_id=row["dataset_id"],
            image_id=row["image_id"],
            image_name=row["image_name"],
            specimen_tag=row["specimen"],
            time_tag=row["time"],
        )
        corr_ctx.append(roi_ctx)

    return corr_ctx


class ProjectDataView:
    def __init__(self, image_contexts: List[ImageContext]):
        self.all_categories = ["image", "roi", "raw_pred", "corrected_pred", "overview"]

        columns = [
            "dataset_id",
            "dataset_name",
            "image_id",
            "image_name",
            "specimen",
            "time",
            "time_tag",
            "class",
        ]

        self.df_all = pd.DataFrame(
            [
                {
                    "dataset_id": ctx.dataset_id,
                    "dataset_name": ctx.dataset_name,
                    "image_id": ctx.image_id,
                    "image_name": ctx.image_name,
                    "specimen": ctx.specimen_tag,
                    "time": ctx.time_idx,
                    "time_tag": ctx.time_tag,
                    "class": ctx.image_class,
                }
                for ctx in image_contexts
            ],
            columns=columns,
        )
        
        df = self.df_all[self.df_all["class"] != "other"].copy()

        df_other = self.df_all[self.df_all["class"] == "other"].copy()
        
        df_summary = self._construct_df_summary(df)

        # Remove rows with an image missing
        image_missing_anomalies = df_summary["image"] == 0
        df, df_other, df_summary, anomalous_image_missing = (
            self._remove_anomalies_from_df(
                df, df_other, df_summary, image_missing_anomalies
            )
        )

        # Remove rows with multiple images
        multiple_images_anomalies = df_summary["image"] > 1
        df, df_other, df_summary, anomalous_multi_image = (
            self._remove_anomalies_from_df(
                df, df_other, df_summary, multiple_images_anomalies
            )
        )

        # Store the dataframe
        self.df = df

        # Image but no roi
        self.roi_missing: List[ImageContext] = _find_roi_missing(df, df_summary)

        # Roi but no preds or corrections
        self.pred_missing: List[ImageContext] = _find_pred_missing(df, df_summary)

        # Preds but no corrections
        self.corr_missing: List[ImageContext] = _find_corr_missing(df, df_summary)

        self.report_data = ReportData(
            n_specimens=self.df["specimen"].nunique(),
            n_times=self.df["time"].nunique(),
            other_files=df_other,
            all_categories=self.all_categories,
            corr_missing_ids=[
                ctx.image_id for ctx in self.corr_missing if ctx.image_id is not None
            ],
            anomalous_multi_image=anomalous_multi_image,
            anomalous_image_missing=anomalous_image_missing,
        )

    @property
    def cases(self) -> List[str]:
        if self.df is not None:
            return self.df["specimen"].unique().tolist()
        else:
            return []

    def _construct_df_summary(self, df: pd.DataFrame) -> pd.DataFrame:
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
        return df_summary

    def _remove_anomalies_from_df(
        self,
        df: pd.DataFrame,
        df_other: pd.DataFrame,
        df_summary: pd.DataFrame,
        summary_filter: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        anomalous_summary = df_summary[summary_filter]

        if anomalous_summary.empty:
            return (
                df,
                df_other,
                df_summary,
                anomalous_summary["specimen"].values.tolist(),
            )

        index_to_remove = df.set_index(["specimen", "time"]).index.isin(
            anomalous_summary.set_index(["specimen", "time"]).index
        )

        df_other = pd.concat([df_other, df[index_to_remove].copy()], ignore_index=True)
        df = df[~index_to_remove].copy()

        df_summary = self._construct_df_summary(df)

        return df, df_other, df_summary, anomalous_summary["specimen"].values.tolist()

    def print_summary(self) -> None:
        # Print a small report
        print("\n" + "=" * 60)
        print(f"📊 Project Summary")
        print("=" * 60)
        print(f"🐭 Cases identified:     {self.report_data.n_specimens}")
        print(f"🕒 Scan times:           {self.report_data.n_times}")
        print("\n⚠️ Warnings:")

        n_images_other = len(self.report_data.other_files)
        if n_images_other > 0:
            print(
                f"  - {n_images_other} files couldn't be identified as one of {self.report_data.all_categories} and will be ignored."
            )

        if len(self.report_data.corr_missing_ids) > 0:
            if len(self.report_data.corr_missing_ids) > 5:
                _ids_to_report = self.report_data.corr_missing_ids[:5]
                corrections_missing_ids = ", ".join(map(str, _ids_to_report))
                extra = len(self.report_data.corr_missing_ids) - 5
                msg = f"{corrections_missing_ids} + {extra} more."
            else:
                msg = ", ".join(map(str, self.report_data.corr_missing_ids))
            print(
                f"  - {len(self.report_data.corr_missing_ids)} corrected masks are missing for these image IDs: {msg}"
            )

        if len(self.report_data.anomalous_multi_image) > 0:
            print(
                f"  - {len(self.report_data.anomalous_multi_image)} specimen-time combinations have multiple matching `image` files and will be ignored."
            )
            print(f"    Occured for: {self.report_data.anomalous_multi_image}")

        if len(self.report_data.anomalous_image_missing) > 0:
            print(
                f"  - {len(self.report_data.anomalous_image_missing)} specimen-time combinations have no matching `image` files and will be ignored."
            )
            print(f"    Occured for: {self.report_data.anomalous_image_missing}")

        if all(
            [
                len(self.report_data.corr_missing_ids) == 0,
                n_images_other == 0,
                len(self.report_data.anomalous_image_missing) == 0,
                len(self.report_data.anomalous_multi_image) == 0,
            ]
        ):
            print("  - No issues found 🎉")
        else:
            if len(self.report_data.other_files) > 0:
                print("\n➡️ OMERO files that were ignored (you may want to check them):")
                print(self.report_data.other_files)

        print("=" * 60 + "\n")

    def images_timeseries_ids(self, specimen_name: str) -> Tuple[List[int], List[int]]:
        """Returns the indeces of the images in a timeseries."""
        image_img_ids = self.df[
            (self.df["specimen"] == specimen_name) & (self.df["class"] == "image")
        ][["image_id", "time"]]
        image_img_ids.sort_values(by="time", ascending=True, inplace=True)

        return image_img_ids["image_id"].tolist(), image_img_ids["time"].tolist()

    def tumor_timeseries_ids(self, specimen: str) -> Tuple[List[int], List[int]]:
        """Returns the indeces of the labeled images in a timeseries. Priority to images with the #corrected tag, otherwise #raw_pred is used."""

        def filter_group(group):
            if "corrected_pred" in group["class"].values:
                return group[group["class"] == "corrected_pred"].iloc[0]
            else:
                return group[group["class"] == "raw_pred"].iloc[0]

        roi_img_ids = self.df[
            (self.df["specimen"] == specimen) & (self.df["class"] == "roi")
        ][["image_id", "time"]]

        labels_img_ids = self.df[
            (self.df["specimen"] == specimen)
            & (self.df["class"].isin(["corrected_pred", "raw_pred"]))
        ][["image_id", "time", "class"]]

        labels_img_ids = labels_img_ids.groupby("time")[labels_img_ids.columns].apply(filter_group).reset_index(drop=True)

        labels_img_ids = pd.merge(
            roi_img_ids,
            labels_img_ids,
            on="time",
            how="left",
            suffixes=("_rois", "_labels"),
        )

        labels_img_ids.sort_values(by="time", ascending=True, inplace=True)

        return (
            labels_img_ids["image_id_rois"].astype("Int64").tolist(),
            labels_img_ids["image_id_labels"].astype("Int64").tolist(),
        )

    def specimen_times(self, specimen_name: str) -> List[str]:
        specimen_df = self.df[self.df["specimen"] == specimen_name]
        return np.unique(specimen_df["time"].tolist()).astype(str).tolist()

    def specimen_image_classes(self, specimen: str, time: str) -> List[str]:
        # Handles '-1.0' which needs to be cast into a float first.
        time = int(float(time))  # type: ignore

        sub_df = self.df[(self.df["specimen"] == specimen) & (self.df["time"] == time)]
        image_classes = sub_df["class"].tolist()

        if ((sub_df["class"] == "roi").sum() > 1) | (
            (sub_df["class"] == "image").sum() > 1
        ):
            print("Duplicate images!")
            image_classes = []
        else:
            n_matches = len(image_classes)
            image_classes = np.unique(image_classes).tolist()
            if len(image_classes) != n_matches:
                print("Warning - Duplicate predictions found!")

        return image_classes

    def image_attribute_from_id(self, image_id: int, attribute: str) -> Any:
        image_df = self.df_all[self.df_all["image_id"] == image_id]
        return image_df[attribute].tolist()[0]

    def complete(self, image_ctx: ImageContext) -> ImageContext:
        if image_ctx.time_idx is None:
            raise RuntimeError("Context needs a scan time.")

        time_int = int(image_ctx.time_idx)

        sub_df = self.df[
            (self.df["specimen"] == image_ctx.specimen_tag)
            & (self.df["time"] == time_int)
            & (self.df["class"] == image_ctx.image_class)
        ]

        image_ctx.image_id = sub_df["image_id"].tolist()[0]
        image_ctx.image_name = sub_df["image_name"].tolist()[0]
        image_ctx.dataset_id = sub_df["dataset_id"].tolist()[0]

        return image_ctx

    def cb_dataset_image_data(self, dataset_id: int) -> Tuple[List[str], List[int]]:
        df_sorted = self.df_all[self.df_all["dataset_id"] == dataset_id].sort_values(
            by="image_id"
        )[["image_id", "image_name"]]
        df_sorted["title"] = df_sorted.apply(
            lambda row: f"{row['image_id']} - {row['image_name']}", axis=1
        )
        titles = df_sorted["title"].tolist()
        image_ids = df_sorted["image_id"].tolist()

        return titles, image_ids

    def dataset_data_and_titles(self) -> Tuple[List[int], List[str]]:
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

        return (dataset_data, dataset_titles)

    def get_dataset_id(self, specimen_name: str) -> int:
        specimen_df = self.df[self.df["specimen"] == specimen_name]
        dataset_ids = specimen_df["dataset_id"].unique()
        if len(dataset_ids) > 1:
            raise RuntimeError(
                f"Multiple datasets found for this specimen name ({specimen_name})."
            )
        return dataset_ids[0]  # It should be the first (and unique) item
