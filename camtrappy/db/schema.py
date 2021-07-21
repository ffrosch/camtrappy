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


Base = declarative_base()

class Video(Base):

    __tablename__ = "videos"

    id = Column(Integer, primary_key=True)
    path = Column(String)
    date = Column(Date)
    starttime = Column(Time)
    duration = Column(Time)
    date_added = Column(DateTime(timezone=True), server_default=func.now())
    location_id = Column(Integer, ForeignKey('locations.id'))
    location = relationship("Location", back_populates="videos")


class Location(Base):

    __tablename__ = "locations"

    id = Column(Integer, primary_key=True)
    name = Column(String)
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
