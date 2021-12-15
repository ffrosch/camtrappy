from abc import ABCMeta, abstractmethod
from typing import Dict

class IProject(metaclass=ABCMeta):
    projectfolder: str
    videoformat: str
    folderstructure: str

    _kwargs: Dict[str, str]
    folderstructure_placeholders: Dict[str, str]

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, 'projectfolder') and
            hasattr(subclass, 'videoformat') and
            hasattr(subclass, 'folderstructure') and
            hasattr(subclass, 'folder_hierarchy') and
            hasattr(subclass, 'folder_indices') and
            hasattr(subclass, 'folder_structure_wildcards') and
            hasattr(subclass, 'folders') and
            hasattr(subclass, 'format_paths') and
            callable(subclass.format_paths) and
            hasattr(subclass, 'get_folder') and
            callable(subclass.get_folder) and
            hasattr(subclass, 'get_videos') and
            callable(subclass.get_videos) or
            NotImplemented
        )

    @property
    @abstractmethod
    def folder_hierarchy(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def folder_indices(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def folder_structure_wildcards(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def folders(self):
        raise NotImplementedError

    @abstractmethod
    def format_paths(self):
        raise NotImplementedError

    @abstractmethod
    def get_folder(self):
        raise NotImplementedError

    @abstractmethod
    def get_videos(self):
        raise NotImplementedError