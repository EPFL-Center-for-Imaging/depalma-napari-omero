import re
import ezomero
import numpy as np
from omero.gateway import BlitzGateway
from omero.gateway import TagAnnotationWrapper

import tempfile
import tifffile
from pathlib import Path

OMERO_HOST = "omero-server.epfl.ch"
OMERO_PORT = 4064
OMERO_GROUP = "imaging-updepalma"

class OmeroServer:
    def __init__(self) -> None:
        self.conn = None

    @property
    def projects(self):
        projects = {}
        for p in self.conn.listProjects():
            projects[str(p.getName())] = int(p.getId())
        return projects
    
    def get_n_datasets_in_project(self, project_id: int):
        project = self.get_project(project_id)
        n_datasets = len(list(project.listChildren()))
        return n_datasets
    
    def project_data_generator(self, project_id):

        project = self.get_project(project_id)

        for dataset in project.listChildren():
            dataset_id = dataset.getId()
            dataset = self.get_dataset(dataset_id)

            for image in dataset.listChildren():
                image_id = image.getId()
                image_name = image.getName()
                image_tags = self.get_image_tags(image_id)

                specimen = self.find_specimen_tag(image_tags)
                time, time_tag = self.find_time_tag(image_tags)
                image_tag_list = self.find_image_tag(image_tags)
                pred_tag_list = self.find_pred_tag(image_tags)

                if (specimen is None) | (time is np.nan):
                    image_class = "other"
                elif len(image_tag_list) >= 1:
                    image_class = "image"
                elif "roi" in image_tags:
                    image_class = "roi"
                elif "corrected" in image_tags:
                    image_class = "corrected_pred"
                elif len(pred_tag_list) >= 1:
                    image_class = "raw_pred"
                else:
                    image_class = "other"
                
                yield (dataset_id, image_id, image_name, specimen, time, time_tag, image_class)

    def login(self, user: str, password: str):
        self.user = user
        self.password = password

    def connect(self) -> bool:
        self.quit()
        self.conn = ezomero.connect(
            user=self.user,
            password=self.password,
            group=OMERO_GROUP,
            host=OMERO_HOST,
            port=OMERO_PORT,
            secure=True,
            config_path=None,
        )

        return self.conn is not None

    def __exit__(self):
        self.quit()

    def __del__(self):
        self.quit()

    def quit(self) -> None:
        if isinstance(self.conn, BlitzGateway):
            self.conn.close()

    def get_project(self, project_id: int):
        return self.conn.getObject("Project", project_id)

    def get_dataset(self, dataset_id: int):
        return self.conn.getObject("Dataset", dataset_id)

    def get_image(self, image_id: int):
        return self.conn.getObject("Image", image_id)

    def get_tag(self, tag_id: int):
        return self.conn.getObject("TagAnnotation", tag_id)

    def get_image_tags(self, image_id: int):
        """Returns a list of tags for a given image ID."""
        image = self.get_image(image_id)
        tags = [
            ann.getTextValue() 
            for ann in image.listAnnotations()
                if isinstance(ann, TagAnnotationWrapper)
        ]
        return tags
    
    def get_image_tag_ids(self, image_id: int):
        image = self.get_image(image_id)
        image_tag_ids = [
            ann.getId() 
            for ann in image.listAnnotations()
                if isinstance(ann, TagAnnotationWrapper)
        ]
        return image_tag_ids
    
    def find_specimen_tag(self, img_tags) -> str:
        """Finds a specimen name (e.g. 'C25065') among image tags based on a regular expression."""
        r = re.compile("C[0-9]+|Animal-[0-9]+")
        specimen_name_tag = list(sorted(filter(r.match, img_tags)))
        if len(specimen_name_tag) == 0:
            return None
        specimen_name_tag = specimen_name_tag[0]

        return specimen_name_tag

    def find_image_tag(self, img_tags) -> list:
        r = re.compile("(I|i)mage(s?)")
        image_tag = list(filter(r.match, img_tags))
        if len(image_tag) == 0:
            return []

        return image_tag
    
    def find_pred_tag(self, img_tags) -> list:
        r = re.compile(".*pred.*")
        pred_tag = list(filter(r.match, img_tags))
        if len(pred_tag) == 0:
            return []

        return pred_tag

    def find_time_tag(self, img_tags) -> int:
        """Finds a time stamp (e.g. 'T2') among image tags based on a regular expression."""
        r = re.compile("(Tm?|SCAN|scan)[0-9]+")
        time_stamp_tag = list(sorted(filter(r.match, img_tags)))
        if len(time_stamp_tag) != 1:
            if len(time_stamp_tag) > 1:
                print('Incoherent scan times: ', time_stamp_tag)
            return (np.nan, np.nan)
        time_stamp_tag = time_stamp_tag[0]

        t = re.findall(r'm?\d+', time_stamp_tag)[0]
        if t == 'm1':
            t = -1
        else:
            t = int(t)

        return (t, time_stamp_tag)
    
    def tag_image_with_tag(self, image_id: int, tag_id: int):
        tag_obj = self.get_tag(tag_id)
        image = self.get_image(image_id)
        image.linkAnnotation(tag_obj)

    def post_image_to_ds(
        self, image: np.ndarray, dataset_id: int, image_title: str = ""
    ) -> int:
        """(Image will note be downloadable) Posts a numpy array as image to dataset."""
        image = np.swapaxes(image, 0, 2)

        while image.ndim < 5:
            image = image[..., np.newaxis]

        posted_img_id = ezomero.post_image(
            self.conn, image, image_title, dataset_id=dataset_id
        )

        return posted_img_id
    
    def import_image_to_ds(
        self, image: np.ndarray, project_id: int, dataset_id: int, image_title: str
    ) -> int:
        with tempfile.NamedTemporaryFile(prefix=f"{Path(image_title).stem}_", suffix='.tif', delete=True) as temp_file:
            # The file name always has a random string attached.
            tifffile.imwrite(temp_file.name, image)

            image_id_list = ezomero.ezimport(
                self.conn, temp_file.name, project=project_id, dataset=dataset_id
            )

            posted_img_id = image_id_list[0]
        
        return posted_img_id

    def download_image(self, image_id: int):
        return np.squeeze(ezomero.get_image(self.conn, image_id)[1])
    
    def delete_image(self, image_id: int):
        self.conn.deleteObjects("Image", [image_id], wait=True)

    def copy_image_tags(self, src_image_id, dst_image_id, exclude_tags=[]):
        src_image_tags = self.get_image_tags(src_image_id)
        src_image_tag_ids = self.get_image_tag_ids(src_image_id)
        for tag_id, tag in zip(src_image_tag_ids, src_image_tags):
            if tag in exclude_tags:
                continue

            self.tag_image_with_tag(dst_image_id, tag_id)