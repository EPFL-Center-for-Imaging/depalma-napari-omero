import re
from typing import List


class TagsProcessor:
    @classmethod
    def get_specimen_tags(cls, tags: List[str]) -> List[str]:
        r = re.compile("C[0-9]+|Animal-[0-9]+")
        return list(sorted(filter(r.match, tags)))

    @classmethod
    def get_image_tags(cls, img_tags: List[str]) -> List[str]:
        r = re.compile("(I|i)mage(s?)")
        return list(filter(r.match, img_tags))

    @classmethod
    def get_raw_pred_tags(cls, img_tags: List[str]) -> List[str]:
        r = re.compile(".*pred.*")
        return list(filter(r.match, img_tags))

    @classmethod
    def get_scan_time_tags(cls, img_tags: List[str]) -> List[str]:
        """Finds a time stamp (e.g. 'T2') among image tags based on a regular expression."""
        r = re.compile("(Tm?|SCAN|scan)[0-9]+")
        # r = re.compile("(Tm?|SCAN|scan)[0-9]+")  # TODO: Should we remove Tm1? = T?
        return list(sorted(filter(r.match, img_tags)))
    
    @classmethod
    def get_scan_time_idx(cls, time_tag: str) -> float:
        t = re.findall(r"m?\d+", time_tag)[0]
        t = -1.0 if t == "m1" else float(t)
        return t
        
