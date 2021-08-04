import json
import os

from sqlalchemy import ForeignKey

from sqlalchemy import (
    Column,
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Time,
    UniqueConstraint
)

from sqlalchemy import and_, select, PickleType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import column_property, relationship
from sqlalchemy.sql import func

from typing import Any, Dict

Base = declarative_base()


class Project(Base):

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    date_created = Column(Date, server_default=func.now())
    date_finished = Column(Date)
    projectfolder = Column(String)
    datafolder = Column(String)
    relative_paths = Column(Boolean)
    description = Column(String)
    locations = relationship("Location", back_populates="project")


class Location(Base):

    __tablename__ = "locations"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    folder = Column(String)
    lat = Column(Float)
    lon = Column(Float)
    videos = relationship("Video", back_populates="location")
    project_id = Column(Integer, ForeignKey('projects.id'))
    project = relationship("Project", back_populates="locations")
    __table_args__ = (UniqueConstraint('name', 'project_id', name='_name_project_uc'),)


class Video(Base):
    # TODO: combine date and time column into datetime column
    # -> this will simplify timezone adjustments
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True)
    path = Column('path', String, nullable=False)
    date = Column(Date)
    time = Column(Time)
    fps = Column(Float)
    duration = Column(Float)
    date_added = Column(DateTime(timezone=True), server_default=func.now())
    location_id = Column(Integer, ForeignKey('locations.id'))
    location = relationship("Location", back_populates="videos")
    data = relationship("VideoObject", back_populates="video")
    Index('video_date_time_idx', 'date', 'time')

    def __repr__(self):
        return f'Video(id={self.id}, path={self.path}, date={self.date}, '\
               f'time={self.time}, fps={self.fps}, duration={self.duration})'

    # greater than (gt) and lower than (lt)
    # for easy chronological sorting, e.g. with sorted()
    def __gt__(self, other):
        return self.date >= other.date and self.time > other.time

    def __lt__(self, other):
        return self.date <= other.date and self.time < other.time

    def to_dict(self) -> Dict[str, Any]:
        """Return Video attributes as dictionary.

        Dict structure is {attribute_name: attribute}.
        """
        return dict(id=self.id, path=self.path, date=self.date, time=self.time,
            fps=self.fps, duration=self.duration)


query_datafolder = select(Project.datafolder).where(
    and_(Project.id == Location.project_id,
         Location.id == Video.location_id)
).scalar_subquery()

query_locationfolder = select(Location.folder).where(
    Location.id == Video.location_id
).scalar_subquery()

Video.datafolder = column_property(query_datafolder)
Video.locationfolder = column_property(query_locationfolder)
Video.fullpath = column_property(Video.datafolder +
                                 os.sep +
                                 Video.locationfolder +
                                 os.sep +
                                 Video.path)


class Object(Base):
    __tablename__ = "objects"

    id = Column(Integer, primary_key=True)
    data = relationship("VideoObject", back_populates="object")



class VideoObject(Base):
    __tablename__ = "videos_objects"

    video_id = Column(Integer, ForeignKey('videos.id'), primary_key=True)
    object_id = Column(Integer, ForeignKey('objects.id'), primary_key=True)
    frames = Column(JSON)
    bboxes = Column(JSON)
    centroids = Column(JSON)
    thumbnail = Column(LargeBinary)
    object = relationship("Object", back_populates="data")
    video = relationship("Video", back_populates="data")
