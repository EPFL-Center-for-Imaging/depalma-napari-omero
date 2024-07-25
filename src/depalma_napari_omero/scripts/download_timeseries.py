import sys
import os
import getpass
from pathlib import Path
import numpy as np
import tifffile

from depalma_napari_omero.omero_server import OmeroServer
from depalma_napari_omero.omero_widget import timeseries_ids, combine_images


def download_project_timeseries(
    server: OmeroServer, project_id: int, output_dir: Path
):
    """Download all available image/labels timeseries in an OMERO project."""
    # Create a directory for the project in the output directory
    output_dir = output_dir / str(project_id)
    if not output_dir.exists():
        os.makedirs(output_dir)
        print(f"Created directory: ", output_dir)

    # Get the project's data summary
    df, df_summary = server.get_project_data(project_id)
    print(df_summary)

    df.to_csv(output_dir / "df.csv")
    df_summary.to_csv(output_dir / "df_summary.csv")

    project_specimens = np.unique(df["specimen"].tolist())
    for specimen_name in project_specimens:
        # Sub-directory for the specimen
        specimen_dir = output_dir / specimen_name
        if not specimen_dir.exists():
            os.mkdir(specimen_dir)
        else:
            print(f"{specimen_dir} already exists. Skipping...")
            continue

        # Get the image IDs of the timeseries
        roi_timeseries_ids, labels_timeseries_ids = timeseries_ids(
            df, specimen_name
        )

        print(f"{roi_timeseries_ids=}")
        print(f"{labels_timeseries_ids=}")

        # Download the ROIs in a single TIF.
        n_rois_timeseries = len(roi_timeseries_ids)
        if n_rois_timeseries == 0:
            print("No ROIs to download.")
        else:
            print(f"Downloading {n_rois_timeseries} ROIs...")
            rois_file = f"{specimen_name}_rois_{n_rois_timeseries}_scans.tif"
            rois_timeseries = combine_images(
                [
                    server.download_image(img_id)
                    for img_id in roi_timeseries_ids
                ]
            )
            tifffile.imwrite(specimen_dir / rois_file, rois_timeseries)
            print(f"Saved {specimen_dir / rois_file}")

        # Download the tumor labels in a single TIF.
        n_nans_labels_timeseries = np.isnan(labels_timeseries_ids).any().sum()
        n_labels_timeseries = (
            len(labels_timeseries_ids) - n_nans_labels_timeseries
        )
        if n_labels_timeseries == 0:
            print("No tumor labels to download.")
        else:
            print(f"Downloading {n_labels_timeseries} tumor labels...")
            labels_file = (
                f"{specimen_name}_labels_{n_rois_timeseries}_scans.tif"
            )
            labels_timeseries = combine_images(
                [
                    server.download_image(img_id)
                    for img_id in labels_timeseries_ids
                ]
            )
            labels_timeseries = labels_timeseries.astype(np.uint16)
            tifffile.imwrite(specimen_dir / labels_file, labels_timeseries)
            print(f"Saved {specimen_dir / labels_file}")


if __name__ == "__main__":
    # Usage: `python download_timeseries.py 512 /to/output/dir/`

    server = OmeroServer()

    _, project_id, output_dir = sys.argv
    project_id = int(project_id)
    output_dir = Path(output_dir)

    print(
        f"Downloading project ID={project_id} into: {str(output_dir.resolve())}"
    )

    user = "imaging-robot"

    server = OmeroServer()
    server.login(user, getpass.getpass(f"OMERO Password for {user}:"))
    conn = server.connect()
    if not conn:
        print("Could not connect to OMERO server.")
    else:
        print(server.projects)
        download_project_timeseries(server, project_id, output_dir)
        print("Done.")

    server.quit()
