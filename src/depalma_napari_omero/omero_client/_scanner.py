from tqdm import tqdm

from depalma_napari_omero.omero_client._client import OmeroClient
from depalma_napari_omero.omero_client._view import ProjectDataView
from depalma_napari_omero.omero_client._tags_processor import TagsProcessor
from depalma_napari_omero.omero_client._context import ImageContext


class ProjectScanner:
    def __init__(
        self,
        omero_client: OmeroClient,
        project_id: int,
        project_name: str,
        launch_scan: bool,
    ):
        self.omero_client = omero_client
        self.id = project_id
        self.name = project_name

        self.image_contexts = []

        if launch_scan:
            self.update()

    @property
    def view(self) -> ProjectDataView:
        return ProjectDataView(self.image_contexts)

    @property
    def n_datasets(self) -> int:
        """Number of datasets in the OMERO project."""
        omero_project = self.omero_client.get_project(self.id)
        return len(list(omero_project.listChildren()))

    def update(self):
        for _ in self.launch_scan():
            continue

    def launch_scan(self):
        self.image_contexts = []
        previous_dataset_id = None
        k = 0
        with tqdm(total=self.n_datasets, desc="Scanning project") as pbar:
            for image_context in self._image_context_generator():
                self.image_contexts.append(image_context)

                if (previous_dataset_id is None) or (
                    previous_dataset_id != image_context.dataset_id
                ):
                    previous_dataset_id = image_context.dataset_id
                    pbar.update(1)
                    k += 1
                    yield k
        
        self.view.print_summary()

    def upload_image(self, image_ctx: ImageContext, image_tag_id: int):
        if image_ctx.project_id is None:
            raise RuntimeError(f"Image upload needs a project ID!")
        
        if image_ctx.image is None:
            raise RuntimeError(f"Image upload needs an image array!")
        
        if image_ctx.image_name is None:
            raise RuntimeError(f"Image upload needs an image name!")
        
        if image_ctx.time_tag is None:
            raise RuntimeError(f"Image upload needs a time tag!")
            
        if image_ctx.specimen_tag is None:
            raise RuntimeError(f"Image upload needs a specimen tag!")
        
        dataset_name = image_ctx.specimen_tag
        if dataset_name is None:
            raise RuntimeError(f"Cannot upload an image in this dataset: {dataset_name}")
        
        if image_ctx.specimen_tag in self.view.cases:
            dataset_id = self.view.get_dataset_id(dataset_name)
        else:
            dataset_id = self.omero_client.post_dataset(image_ctx.project_id, dataset_name)
        
        image_ctx.dataset_id = dataset_id
        image_ctx.dataset_name = dataset_name

        # Post the image in the dataset
        posted_image_id = self.omero_client.import_image_to_ds(
            image_ctx.image,
            image_ctx.project_id,
            image_ctx.dataset_id,
            image_ctx.image_name,
        )
        
        image_ctx.image_id = posted_image_id

        # Tag the image appropriately
        scan_time_tag_id = self.omero_client.create_tag(image_ctx.project_id, image_ctx.time_tag)
        specimen_tag_id = self.omero_client.create_tag(image_ctx.project_id, image_ctx.specimen_tag)
        project_tag_id = self.omero_client.create_tag(image_ctx.project_id, self.name)

        self.omero_client.tag_image_with_tag(posted_image_id, tag_id=image_tag_id)
        self.omero_client.tag_image_with_tag(posted_image_id, tag_id=scan_time_tag_id)
        self.omero_client.tag_image_with_tag(posted_image_id, tag_id=specimen_tag_id)
        self.omero_client.tag_image_with_tag(posted_image_id, tag_id=project_tag_id)

        self.update()

    def _image_context_generator(self):
        """Iterate over all datasets and images of an OMERO project, and yield an ImageContext."""
        omero_project = self.omero_client.get_project(self.id)
        for dataset in omero_project.listChildren():
            dataset_id = dataset.getId()
            dataset_name = dataset.getName()
            for image in dataset.listChildren():
                image_id = image.getId()
                image_name = image.getName()
                image_tags = self.omero_client.get_image_tags(image_id)

                # Process specimen tags
                specimen_tags = TagsProcessor.get_specimen_tags(image_tags)
                if len(specimen_tags) == 0:
                    specimen_tag = None
                elif len(specimen_tags) >= 1:
                    if len(specimen_tags) > 1:
                        print(f"Multiple specimen name tags found: {specimen_tags} among {image_tags} ({image_id=}). Will use: {specimen_tags[0]}")
                    specimen_tag = specimen_tags[0]

                # Process time tags
                time_tags = TagsProcessor.get_scan_time_tags(image_tags)
                if len(time_tags) == 0:
                    time_idx = None
                    time_tag = None
                elif len(time_tags) >= 1:
                    if len(time_tags) > 1:
                        print(f"Incoherent scan times: {time_tags} ({image_id=}). Will use: {time_tags[0]}.")
                    time_tag = time_tags[0]
                    time_idx = TagsProcessor.get_scan_time_idx(time_tag)

                # Process image class
                image_class = "other"
                if len(TagsProcessor.get_image_tags(image_tags)) >= 1:
                    image_class = "image"
                elif "roi" in image_tags:
                    image_class = "roi"
                elif ("corrected" in image_tags) | ("corrected_pred" in image_tags):
                    image_class = "corrected_pred"
                elif len(TagsProcessor.get_raw_pred_tags(image_tags)) >= 1:
                    image_class = "raw_pred"
                elif "overview" in image_tags:
                    image_class = "overview"

                yield ImageContext(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    image_id=image_id,
                    image_name=image_name,
                    specimen_tag=specimen_tag,
                    time_idx=time_idx,
                    time_tag=time_tag,
                    image_class=image_class,
                )
