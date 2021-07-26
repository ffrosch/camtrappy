from sqlalchemy import ForeignKey

from sqlalchemy import (
    Column,
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Time,
    UniqueConstraint
)

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

from typing import Any, Dict

Base = declarative_base()

class Video(Base):

    __tablename__ = "videos"

    id = Column(Integer, primary_key=True)
    path = Column(String, nullable=False)
    date = Column(Date)
    time = Column(Time)
    fps = Column(Float)
    duration = Column(Float)
    date_added = Column(DateTime(timezone=True), server_default=func.now())
    location_id = Column(Integer, ForeignKey('locations.id'))
    location = relationship("Location", back_populates="videos")

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
