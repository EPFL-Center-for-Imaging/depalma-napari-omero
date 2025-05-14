from typing import List, Literal, Type
import re
import questionary
import uvicorn
from pydantic import BaseModel, Field

from imaging_server_kit import AlgorithmServer

from mousetumorpy import (
    LungsPredictor,
    TumorPredictor,
    run_tracking,
    combine_images,
    to_formatted_df,
)

from depalma_napari_omero.omero_client import OmeroClient
from depalma_napari_omero.omero_client.omero_config import (
    OMERO_HOST,
    OMERO_PORT,
    OMERO_GROUP,
)


class Parameters(BaseModel):
    """Defines the algorithm parameters"""

    workflow_step: Literal["roi", "pred", "track"] = Field(
        default="roi",
        title="Workflow step",
        description="The workflow step to perform.",
        json_schema_extra={"widget_type": "dropdown"},
    )
    lungs_model: Literal["v1"] = Field(
        default="v1",
        title="Lungs model",
        description="The model used for lungs segmentation.",
        json_schema_extra={"widget_type": "dropdown"},
    )
    tumor_model: Literal["v4", "oct24"] = Field(
        default="oct24",
        title="Tumor model",
        description="The model used for tumor segmentation.",
        json_schema_extra={"widget_type": "dropdown"},
    )
    image_id: int = Field(
        default=1,
        title="Image ID",
        description="The image ID.",
        ge=1,
        le=65_536,
        json_schema_extra={
            "widget_type": "int",
            "step": 1,
        },
    )
    dataset_id: int = Field(
        default=1,
        title="Dataset ID",
        description="The dataset ID.",
        ge=1,
        le=65_536,
        json_schema_extra={
            "widget_type": "int",
            "step": 1,
        },
    )
    project_id: int = Field(
        default=1,
        title="Project ID",
        description="The project ID.",
        ge=1,
        le=65_536,
        json_schema_extra={
            "widget_type": "int",
            "step": 1,
        },
    )
    posted_image_name: str = Field(
        default="image_name",
        title="Image name",
        description="Posted image name.",
        json_schema_extra={
            "widget_type": "str",
        },
    )
    roi_timeseries_ids: List[int] = Field(
        default="",
        title="ROI timeseries IDs",
        description="List of ROI ids in the timeseries.",
        json_schema_extra={
            "widget_type": "str",
        },
    )
    tumor_timeseries_ids: List[int] = Field(
        default="",
        title="Rumor timeseries IDs",
        description="List of tumor ids in the timeseries.",
        json_schema_extra={
            "widget_type": "str",
        },
    )


def find_image_tag(img_tags) -> list:
    r = re.compile("(I|i)mage(s?)")
    image_tag = list(filter(r.match, img_tags))
    if len(image_tag) == 0:
        return []

    return image_tag


class MouseTumorComputeServer(AlgorithmServer):
    def __init__(
        self,
        host: str,
        group: str,
        port: int,
        algorithm_name: str = "mousetumorpy",
        parameters_model: Type[BaseModel] = Parameters,
    ):
        super().__init__(algorithm_name, parameters_model)

        self.omero_client = OmeroClient(
            host=host,
            group=group,
            port=port,
        )

    def login(self, user, password):
        self.omero_client.login(
            user=user,
            password=password,
        )

    def run_algorithm(
        self,
        workflow_step: str,
        image_id: int,
        dataset_id: int = None,
        project_id: int = None,
        lungs_model: str = None,
        tumor_model: str = None,
        posted_image_name: str = None,
        roi_timeseries_ids: List[int] = None,
        tumor_timeseries_ids: List[int] = None,
        **kwargs,
    ) -> List[tuple]:
        """Run the selected workflow step."""
        if workflow_step == "roi":
            status_code = self._compute_roi(
                lungs_model, posted_image_name, image_id, dataset_id, project_id
            )
        elif workflow_step == "pred":
            status_code = self._compute_nnunet(
                tumor_model, posted_image_name, image_id, dataset_id, project_id
            )
        elif workflow_step == "track":
            status_code = self._compute_tracking(
                image_id, roi_timeseries_ids, tumor_timeseries_ids
            )

        return [("Workflow completed!", {"status": status_code}, "text")]

    def _compute_roi(
        self, model, posted_image_name, image_id, dataset_id, project_id
    ) -> None:
        predictor = LungsPredictor(model)

        image = self.omero_client.download_image(image_id)

        try:
            roi, lungs_roi = predictor.compute_3d_roi(image)
        except:
            print(
                f"An error occured while computing the ROI in this image: ID={image_id}. Skipping..."
            )
            return -1

        posted_image_id = self.omero_client.import_image_to_ds(
            roi, project_id, dataset_id, posted_image_name
        )

        # Upload the lungs as omero ROI
        self.omero_client.post_binary_mask_as_roi(posted_image_id, lungs_roi)

        # Add tags
        roi_tag_id = self.omero_client.create_tag_if_not_exists(project_id, "roi")
        self.omero_client.tag_image_with_tag(posted_image_id, tag_id=roi_tag_id)

        image_tags_list = find_image_tag(self.omero_client.get_image_tags(image_id))

        self.omero_client.copy_image_tags(
            src_image_id=image_id,
            dst_image_id=posted_image_id,
            exclude_tags=image_tags_list,
        )

        return 2

    def _compute_nnunet(
        self, model, posted_image_name, image_id, dataset_id, project_id
    ) -> None:
        predictor = TumorPredictor(model)

        image = self.omero_client.download_image(image_id)

        try:
            image_pred = predictor.predict(image)
        except:
            print(
                f"An error occured while computing the NNUNET prediction in this image: ID={image_id}."
            )
            return -1

        posted_image_id = self.omero_client.import_image_to_ds(
            image_pred, project_id, dataset_id, posted_image_name
        )

        pred_tag_id = self.omero_client.create_tag_if_not_exists(project_id, "raw_pred")
        self.omero_client.tag_image_with_tag(posted_image_id, tag_id=pred_tag_id)
        self.omero_client.copy_image_tags(
            src_image_id=image_id,
            dst_image_id=posted_image_id,
            exclude_tags=["roi"],
        )

        return 2

    def _compute_tracking(
        self, image_id, roi_timeseries_ids, tumor_timeseries_ids
    ) -> None:
        rois_timeseries_list = []
        lungs_timeseries_list = []
        for roi_id in roi_timeseries_ids:
            image = self.omero_client.download_image(roi_id)
            rois_timeseries_list.append(image)
            lungs = self.omero_client.download_binary_mask_from_image_rois(roi_id)
            lungs_timeseries_list.append(lungs)

        tumor_timeseries_list = []
        for tumor_id in tumor_timeseries_ids:
            tumor = self.omero_client.download_image(tumor_id)
            tumor_timeseries_list.append(tumor)

        rois_timeseries = combine_images(rois_timeseries_list)
        lungs_timeseries = combine_images(lungs_timeseries_list)
        tumor_timeseries = combine_images(tumor_timeseries_list)

        linkage_df = run_tracking(
            tumor_timeseries,
            rois_timeseries,
            lungs_timeseries,
            with_lungs_registration=True,
            method="laptrack",
            max_dist_px=30,
            dist_weight_ratio=0.9,
            max_volume_diff_rel=1.0,
            memory=0,
        )

        formatted_df = to_formatted_df(linkage_df)

        self.omero_client.attach_table_to_image(
            table=formatted_df,
            image_id=image_id,
        )

        return 2


if __name__ == "__main__":
    user = questionary.text("OMERO username:").ask()
    password = questionary.password("OMERO password:").ask()

    server = MouseTumorComputeServer(
        host=OMERO_HOST,
        group=OMERO_GROUP,
        port=OMERO_PORT,
    )

    server.login(user, password)

    uvicorn.run(server.app, host="0.0.0.0", port=8000)
