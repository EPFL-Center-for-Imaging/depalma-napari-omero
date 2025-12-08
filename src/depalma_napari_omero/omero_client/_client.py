import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

import ezomero
import geojson
import numpy as np
import pandas as pd
import pooch
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from ezomero.rois import Polygon
from omero.gateway import (
    BlitzGateway,
    FileAnnotationWrapper,
    TagAnnotationWrapper,
    _ProjectWrapper,
    _DatasetWrapper,
    _ImageWrapper,
    _RoiWrapper,
)
from skimage.exposure import rescale_intensity

from imaging_server_kit.types._mask import mask2features, features2instance_mask_3d

from depalma_napari_omero.omero_client.omero_config import OmeroConfig


def require_active_conn(func: Callable):
    """Ensure OMERO connection is alive before running `func`."""

    def wrapper(self, *args, **kwargs):
        if self.conn is None:
            self.connect()
        elif isinstance(self.conn, BlitzGateway):
            conn = self.conn.keepAlive()
            if conn is False:
                self.connect()
        return func(self, *args, **kwargs)

    return wrapper


class OmeroClient:
    def __init__(
        self, user: str, password: str, omero_cfg: Optional[OmeroConfig] = None
    ) -> None:
        self.omero_cfg = omero_cfg if omero_cfg is not None else OmeroConfig()
        self.user = user
        self.password = password
        self._conn = None

    @property
    @require_active_conn
    def projects(self) -> Dict[str, int]:
        """Return a mapping of project name to project ID."""
        projects = {}
        for p in self.conn.listProjects():  # type: ignore
            projects[str(p.getName())] = int(p.getId())
        return projects

    @property
    def conn(self) -> Optional[BlitzGateway]:
        return self._conn

    @conn.setter
    def conn(self, val: Optional[BlitzGateway]) -> None:
        self._conn = val

    def connect(self) -> bool:
        self.quit()
        try:
            self.conn = ezomero.connect(
                user=self.user,
                password=self.password,
                group=self.omero_cfg.group,
                host=self.omero_cfg.host,
                port=self.omero_cfg.port,
                secure=True,
                config_path=None,
            )
        except Exception as e:
            self.conn = None

        return isinstance(self.conn, BlitzGateway)

    def __exit__(self):
        self.quit()

    def __del__(self):
        self.quit()

    def quit(self) -> None:
        if isinstance(self.conn, BlitzGateway):
            self.conn.close()

    @require_active_conn
    def get_project(self, project_id: int) -> _ProjectWrapper:
        obj = self.conn.getObject("Project", project_id)  # type: ignore
        if obj is None:
            raise LookupError(f"Project with ID {project_id} was not found on OMERO.")
        return obj

    @require_active_conn
    def get_dataset(self, dataset_id: int) -> _DatasetWrapper:
        obj = self.conn.getObject("Dataset", dataset_id)  # type: ignore
        if obj is None:
            raise LookupError(f"Dataset with ID {dataset_id} was not found on OMERO.")
        return obj

    @require_active_conn
    def get_image(self, image_id: int) -> _ImageWrapper:
        obj = self.conn.getObject("Image", image_id)  # type: ignore
        if obj is None:
            raise LookupError(f"Image with ID {image_id} was not found on OMERO.")
        return obj

    @require_active_conn
    def get_roi(self, roi_id: int) -> _RoiWrapper:
        return self.conn.getObject("ROI", roi_id)  # type: ignore

    @require_active_conn
    def get_shape(self, shape_id: int):
        return ezomero.get_shape(self.conn, shape_id)  # type: ignore

    @require_active_conn
    def get_table(self, table_id: int) -> pd.DataFrame:
        obj = ezomero.get_table(self.conn, table_id)  # type: ignore
        if obj is None:
            raise LookupError(f"Table with ID {table_id} was not found on OMERO.")
        return obj

    @require_active_conn
    def get_tag(self, tag_id: int):
        return self.conn.getObject("TagAnnotation", tag_id)  # type: ignore

    @require_active_conn
    def get_image_tags(self, image_id: int) -> List[str]:
        """Returns a list of tags associated to an image ID."""
        image_tags = []
        for ann in self.get_image(image_id).listAnnotations():
            if isinstance(ann, TagAnnotationWrapper):
                ann_text = ann.getTextValue()
                image_tags.append(ann_text)
        return image_tags

    @require_active_conn
    def get_image_tag_ids(self, image_id: int) -> List[Optional[int]]:
        return [
            ann.getId()
            for ann in self.get_image(image_id).listAnnotations()
            if isinstance(ann, TagAnnotationWrapper)
        ]

    @require_active_conn
    def get_image_table_ids(self, image_id: int) -> List[int]:
        table_ids = []
        image_annotations = self.get_image(image_id).listAnnotations()
        for ann in image_annotations:
            if isinstance(ann, FileAnnotationWrapper):
                ann_id = ann.getId()
                table_ids.append(ann_id)
        return table_ids

    @require_active_conn
    def import_image_to_ds(
        self, image: np.ndarray, project_id: int, dataset_id: int, image_name: str
    ) -> int:
        cache_dir = pooch.os_cache("depalma-napari-omero")
        if not cache_dir.exists():
            os.makedirs(cache_dir)

        with tempfile.NamedTemporaryFile(
            prefix=f"{Path(image_name).stem}_",
            suffix=".ome.tif",
            delete=False,
            dir=cache_dir,
        ) as temp_file:
            file_name = Path(temp_file.name).with_name(
                f"{Path(image_name).stem}.ome.tif"
            )
            OmeTiffWriter.save(image, file_name, dim_order="ZYX")

        temp_file.close()

        image_id_list = ezomero.ezimport(
            self.conn,  # type: ignore
            target=str(file_name),
            project=project_id,
            dataset=dataset_id,
        )

        os.unlink(file_name)

        if image_id_list is not None:
            posted_img_id = image_id_list[0]
            return posted_img_id
        else:
            raise RuntimeError(
                f"An error occurred while importing an image to omero (name: {image_name} ; {project_id=} ; {dataset_id=})"
            )

    @require_active_conn
    def download_image(self, image_id: int) -> np.ndarray:
        ez_image = ezomero.get_image(self.conn, image_id)[1]  # type: ignore
        if ez_image is not None:
            return np.squeeze(ez_image)
        else:
            raise LookupError(
                f"Image with ID {image_id} could not be downloaded from OMERO."
            )

    @require_active_conn
    def delete_image(self, image_id: int) -> None:
        self.conn.deleteObjects("Image", [image_id], wait=True)  # type: ignore

    @require_active_conn
    def tag_image_with_tag(self, image_id: int, tag_id: int) -> None:
        tag_obj = self.get_tag(tag_id)
        image = self.get_image(image_id)
        image.linkAnnotation(tag_obj)  # type: ignore

    @require_active_conn
    def copy_image_tags(self, src_image_id: int, dst_image_id: int, exclude_tags=None):
        if exclude_tags is None:
            exclude_tags = []
        src_image_tags = self.get_image_tags(src_image_id)
        src_image_tag_ids = self.get_image_tag_ids(src_image_id)
        for tag_id, tag in zip(src_image_tag_ids, src_image_tags):
            if tag in exclude_tags:
                continue
            self.tag_image_with_tag(dst_image_id, tag_id)  # type: ignore

    @require_active_conn
    def get_image_rois(self, image_id: int):
        return ezomero.get_roi_ids(self.conn, image_id=image_id)  # type: ignore

    @require_active_conn
    def get_roi_shapes(self, roi_id: int) -> List[int]:
        return ezomero.get_shape_ids(self.conn, roi_id=roi_id)  # type: ignore

    @require_active_conn
    def attach_table_to_image(
        self, table: pd.DataFrame, image_id: int, table_title: str = "Tracking results"
    ) -> int:
        return ezomero.post_table(self.conn, table, "Image", image_id, table_title)  # type: ignore

    @require_active_conn
    def post_dataset(self, project_id: int, dataset_name: str) -> int:
        dataset_id = ezomero.post_dataset(self.conn, dataset_name, project_id)  # type: ignore
        if dataset_id is not None:
            return int(dataset_id)
        else:
            raise RuntimeError(f"Could not post this dataset ({dataset_name=}) in project ID {project_id}.")

    @require_active_conn
    def post_tag_by_name(self, project_id: int, tag_name: str) -> int:
        project = self.get_project(project_id)
        tag_obj = TagAnnotationWrapper(self.conn)
        tag_obj.createAndLink(project, ns=None, val=tag_name)
        tag_id = self.get_tag_id_by_name(project_id, tag_name)
        if tag_id is not None:
            return tag_id
        else:
            raise RuntimeError(
                f"Could not create or retreive tag {tag_name} in project with ID {project_id} on OMERO."
            )

    @require_active_conn
    def get_tag_id_by_name(self, project_id: int, tag_name: str) -> Optional[int]:
        proj_tag_ids = ezomero.get_tag_ids(
            self.conn, "Project", project_id, across_groups=False
        )  # type: ignore
        for tag_id in proj_tag_ids:
            if self.get_tag(tag_id).getTextValue() == tag_name:  # type: ignore
                return tag_id

    @require_active_conn
    def post_roi(self, image_id: int, shapes: List) -> int:
        return ezomero.post_roi(self.conn, image_id, shapes)  # type: ignore

    @require_active_conn
    def post_binary_mask_as_roi(self, image_id: int, mask: np.ndarray) -> int:
        mask = rescale_intensity(mask, out_range=(0, 1)).astype(np.uint8)  # type: ignore

        all_rois = []
        for z_idx, lung_slice in enumerate(mask):
            polygons = mask2features(lung_slice)
            for polygon in polygons:
                points = polygon["geometry"]["coordinates"][0]
                points_ezomero = [(x, y) for x, y in points]
                roi = Polygon(points=points_ezomero, z=z_idx)
                all_rois.append(roi)

        roi_id = self.post_roi(image_id, all_rois)

        return roi_id

    @require_active_conn
    def download_binary_mask_from_image_rois(self, image_id) -> np.ndarray:
        all_roi_ids = self.get_image_rois(image_id)
        image = self.get_image(image_id)

        # Workaround - For images that were not imported as OME-TIFF, the Z dimension is interpreted as T
        size_z = image.getSizeZ()
        if size_z == 1:
            size_z = image.getSizeT()

        img_shape = (
            size_z,
            image.getSizeY(),
            image.getSizeX(),
        )

        features = []
        for detection_id, roi_id in enumerate(all_roi_ids, start=1):
            roi_shape_ids = self.get_roi_shapes(roi_id=roi_id)
            for shape_id in roi_shape_ids:  # Different Z
                geometry = self.get_shape(shape_id=shape_id)
                z_idx = geometry.z
                coords = geometry.points  # List of tuples (x, y)
                coords = np.array(coords)
                coords = np.vstack([coords, coords[0]])  # Close the polygon for QuPath
                geom = geojson.Polygon(coordinates=[coords.tolist()])
                feature = geojson.Feature(
                    geometry=geom,
                    properties={
                        "Detection ID": detection_id,
                        "Class": 1,
                        "z_idx": z_idx,
                    },
                )
                features.append(feature)

        mask = features2instance_mask_3d(features, img_shape)

        return mask

    @require_active_conn
    def create_tag(self, project_id: int, tag: str) -> int:
        """Create a tag for a project if it doesn't exist yet."""
        tag_id = self.get_tag_id_by_name(project_id, tag)
        if tag_id is None:
            tag_id = self.post_tag_by_name(project_id, tag)
        return tag_id
