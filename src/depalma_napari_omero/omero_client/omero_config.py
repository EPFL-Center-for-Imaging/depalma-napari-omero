from dataclasses import dataclass

@dataclass
class OmeroConfig:
    port: int = 4064
    host: str = "omero-server.epfl.ch"
    group: str = "imaging-updepalma"
    default_user: str = "imaging-robot"