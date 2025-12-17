from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import skimage.io
from depalma_napari_omero.omero_client._context import ImageContext
from depalma_napari_omero.omero_client._tags_processor import TagsProcessor


def save_merged_csv(out_dir: Path, out_csv_path: Path) -> None:
    all_dfs = []
    for csv_file in list(out_dir.rglob("*.csv")):
        specimen = csv_file.stem.split("_")[0]

        if TagsProcessor.get_specimen_tags([specimen]) is None:
            raise ValueError(f"Specimen name {specimen} does not comply with naming convention (C*****).")

        df_specimen = pd.read_csv(csv_file, index_col=0)
        df_specimen["Mouse_ID"] = specimen
        all_dfs.append(df_specimen)

    if len(all_dfs) == 0:
        return
    
    merged_df = pd.concat(all_dfs, ignore_index=True)

    columns = ["Mouse_ID"] + [col for col in merged_df.columns if col != "Mouse_ID"]
    merged_df = merged_df[columns]
    merged_df.to_csv(out_csv_path)


def load_ct_from_folder(image_dir: Union[str, Path]) -> ImageContext:
    """Creates a 3D TIFF from a series of 2D tiffs in a folder, according to lab conventions."""
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
    exp_name, scan_time, specimen = (
        Path(tiff_image_files[0]).stem.split("~")[0].split("_")
    )
    
    # Check the naming conventions
    if TagsProcessor.get_specimen_tags([specimen]) is None:
        raise ValueError(
            f"Specimen name {specimen} does not comply with naming convention (C*****)."
        )

    if not isinstance(TagsProcessor.get_scan_time_tags([scan_time])[0], int):
        raise ValueError(
            f"Scan time {scan_time} does not comply with naming convention (SCAN* or T*)."
        )
        
    time_idx = TagsProcessor.get_scan_time_idx(scan_time)

    # Read files into a 3D tiff
    image = np.array([skimage.io.imread(file) for file in tiff_image_files])
    
    image_ctx = ImageContext(
        image_class="image",
        image=image,
        time_tag=scan_time,
        time_idx=time_idx,
        specimen_tag=specimen,
    )
    
    return image_ctx