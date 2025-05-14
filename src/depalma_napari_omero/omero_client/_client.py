import os
import tempfile
from pathlib import Path
from typing import List

import ezomero
import geojson
import numpy as np
import pandas as pd
import pooch
from aicsimageio.writers import OmeTiffWriter
from ezomero.rois import Polygon
from omero.gateway import (BlitzGateway, FileAnnotationWrapper,
                           TagAnnotationWrapper)
from skimage.exposure import rescale_intensity

from imaging_server_kit.core import mask2features, features2instance_mask_3d

from depalma_napari_omero.omero_client.omero_config import (OMERO_GROUP,
                                                            OMERO_HOST,
                                                            OMERO_PORT)


def require_active_conn(func):
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
    def __init__(self, host=None, group=None, port=None) -> None:
        self.conn = None
        if host is None:
            self.host = OMERO_HOST
        else:
            self.host = host
        if group is None:
            self.group = OMERO_GROUP
        else:
            self.group = group
        if port is None:
            self.port = OMERO_PORT
        else:
            self.port = port

    @property
    @require_active_conn
    def projects(self):
        projects = {}
        for p in self.conn.listProjects():
            projects[str(p.getName())] = int(p.getId())
        return projects

    def login(self, user: str, password: str):
        self.user = user
        self.password = password

    def connect(self) -> bool:
        self.quit()
        try:
            self.conn = ezomero.connect(
                user=self.user,
                password=self.password,
                group=self.group,
                host=self.host,
                port=self.port,
                secure=True,
                config_path=None,
            )
        except Exception as e:
            self.conn = None

        return self.conn is not None

    def __exit__(self):
        self.quit()

    def __del__(self):
        self.quit()

    def quit(self) -> None:
        if isinstance(self.conn, BlitzGateway):
            self.conn.close()

    @require_active_conn
    def get_project(self, project_id: int):
        return self.conn.getObject("Project", project_id)

    @require_active_conn
    def get_dataset(self, dataset_id: int):
        return self.conn.getObject("Dataset", dataset_id)

    @require_active_conn
    def get_image(self, image_id: int):
        return self.conn.getObject("Image", image_id)

    @require_active_conn
    def get_roi(self, roi_id: int):
        return self.conn.getObject("ROI", roi_id)

    @require_active_conn
    def get_shape(self, shape_id: int):
        return ezomero.get_shape(self.conn, shape_id)

    @require_active_conn
    def get_table(self, table_id: int):
        return ezomero.get_table(self.conn, table_id)

    @require_active_conn
    def get_tag(self, tag_id: int):
        return self.conn.getObject("TagAnnotation", tag_id)

    @require_active_conn
    def get_image_tags(self, image_id: int):
        """Returns a list of tags for a given image ID."""
        image = self.get_image(image_id)
        tags = [
            ann.getTextValue()
            for ann in image.listAnnotations()
            if isinstance(ann, TagAnnotationWrapper)
        ]
        return tags

    @require_active_conn
    def get_image_tag_ids(self, image_id: int):
        image = self.get_image(image_id)
        image_tag_ids = [
            ann.getId()
            for ann in image.listAnnotations()
            if isinstance(ann, TagAnnotationWrapper)
        ]
        return image_tag_ids

    @require_active_conn
    def get_image_table_ids(self, image_id: int):
        image = self.get_image(image_id)
        table_ids = [
            ann.getId()
            for ann in image.listAnnotations()
            if isinstance(ann, FileAnnotationWrapper)
        ]
        return table_ids

    @require_active_conn
    def import_image_to_ds(
        self, image: np.ndarray, project_id: int, dataset_id: int, image_title: str
    ) -> int:
        cache_dir = pooch.os_cache("depalma-napari-omero")
        if not cache_dir.exists():
            os.makedirs(cache_dir)

        with tempfile.NamedTemporaryFile(
            prefix=f"{Path(image_title).stem}_",
            suffix=".ome.tif",
            delete=False,
            dir=cache_dir,
        ) as temp_file:
            # Remove the random strings attached to the file name
            file_name = temp_file.name.split(".ome.tif")[0][:-9] + ".ome.tif"
            OmeTiffWriter.save(image, file_name, dim_order="ZYX")

            image_id_list = ezomero.ezimport(
                self.conn,
                file_name,
                project=project_id,
                dataset=dataset_id,
            )
            posted_img_id = image_id_list[0]

            temp_file.close()
            os.unlink(file_name)

        return posted_img_id

    @require_active_conn
    def download_image(self, image_id: int):
        return np.squeeze(ezomero.get_image(self.conn, image_id)[1])

    @require_active_conn
    def delete_image(self, image_id: int):
        self.conn.deleteObjects("Image", [image_id], wait=True)

    @require_active_conn
    def tag_image_with_tag(self, image_id: int, tag_id: int):
        tag_obj = self.get_tag(tag_id)
        image = self.get_image(image_id)
        image.linkAnnotation(tag_obj)

    @require_active_conn
    def copy_image_tags(self, src_image_id: int, dst_image_id: int, exclude_tags=[]):
        src_image_tags = self.get_image_tags(src_image_id)
        src_image_tag_ids = self.get_image_tag_ids(src_image_id)
        for tag_id, tag in zip(src_image_tag_ids, src_image_tags):
            if tag in exclude_tags:
                continue
            self.tag_image_with_tag(dst_image_id, tag_id)

    @require_active_conn
    def get_image_rois(self, image_id: int):
        return ezomero.get_roi_ids(
            self.conn,
            image_id=image_id,
        )

    @require_active_conn
    def get_roi_shapes(self, roi_id: int):
        shape_ids = ezomero.get_shape_ids(
            self.conn,
            roi_id=roi_id,
        )
        return shape_ids

    @require_active_conn
    def attach_table_to_image(
        self, table: pd.DataFrame, image_id: int, table_title: str = "Tracking results"
    ):
        table_id = ezomero.post_table(
            conn=self.conn,
            table=table,
            object_type="Image",
            object_id=image_id,
            title=table_title,
        )
        return table_id

    @require_active_conn
    def post_dataset(self, project_id: int, dataset_name: str):
        dataset_id = ezomero.post_dataset(
            conn=self.conn,
            dataset_name=dataset_name,
            project_id=project_id,
        )
        return dataset_id

    @require_active_conn
    def post_tag_by_name(self, project_id: int, tag_name: str):
        project = self.get_project(project_id)
        tag_obj = TagAnnotationWrapper(self.conn)

        tag_obj.createAndLink(
            project,
            ns=None,
            val=tag_name,
        )

        tag_id = self.get_tag_by_name(project_id, tag_name)

        return tag_id

    @require_active_conn
    def get_tag_by_name(self, project_id: int, tag_name: str):
        project_tag_ids = ezomero.get_tag_ids(
            conn=self.conn,
            object_type="Project",
            object_id=project_id,
            across_groups=False,
        )

        for tag_id in project_tag_ids:
            tag = self.get_tag(tag_id)
            selected_tag_name = tag.getTextValue()
            if selected_tag_name == tag_name:
                return tag_id

    @require_active_conn
    def post_roi(self, image_id: int, shapes: List):
        roi_id = ezomero.post_roi(
            conn=self.conn,
            image_id=image_id,
            shapes=shapes,
        )
        return roi_id

    @require_active_conn
    def post_binary_mask_as_roi(self, image_id: int, mask: np.ndarray):
        mask = rescale_intensity(mask, out_range=(0, 1)).astype(np.uint8)

        all_rois = []
        for z_idx, lung_slice in enumerate(mask):
            polygons = mask2features(lung_slice)
            for polygon in polygons:
                points = polygon.get("geometry").get("coordinates")[0]
                points_ezomero = []
                for x, y in points:
                    points_ezomero.append((x, y))
                roi = Polygon(
                    points=points_ezomero,
                    z=z_idx,
                )
                all_rois.append(roi)

        roi_id = self.post_roi(image_id, all_rois)

        return roi_id

    @require_active_conn
    def download_binary_mask_from_image_rois(self, image_id):
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
    def create_tag_if_not_exists(self, project_id: int, tag_name: str):
        tag_id = self.get_tag_by_name(project_id, tag_name)
        if tag_id is None:
            tag_id = self.post_tag_by_name(project_id, tag_name)
        return tag_id
