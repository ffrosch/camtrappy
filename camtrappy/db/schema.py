from sqlalchemy import ForeignKey

from sqlalchemy import Column, Date, DateTime, Float, Integer, String
from sqlalchemy.orm import relation, relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.sqltypes import Date


Base = declarative_base()

class Video(Base):

    __tablename__ = "videos"

    id = Column(Integer, primary_key=True)
    path = Column(String)
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


class Project(Base):

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    date_created = Column(Date, server_default=func.now())
    date_finished = Column(Date)
    locations = relationship("Location", back_populates="project")
