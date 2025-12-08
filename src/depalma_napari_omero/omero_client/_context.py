from dataclasses import dataclass
from typing import List, Optional

import numpy as np

@dataclass
class ImageContext:
    image_class: str
    project_id: Optional[int] = None
    dataset_id: Optional[int] = None
    dataset_name: Optional[str] = None
    image_id: Optional[int] = None
    image_name: Optional[str] = None
    specimen_tag: Optional[str] = None
    time_idx: Optional[float] = None
    time_tag: Optional[str] = None
    image: Optional[np.ndarray] = None
    original_image_id: Optional[int] = None


@dataclass
class SpecimenContext:
    name: str
    times: List[str]
    n_rois: int
    n_labels: int
    n_lungs: int
    n_tracked: int
    roi_series: List[int]
    tumor_series: List[int]
    tracking_table_id: Optional[int] = None
