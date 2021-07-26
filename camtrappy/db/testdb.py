from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, drop_database

from camtrappy.db.schema import Base
from camtrappy.db.utils import new_project
from camtrappy.io.parsing import ProjectParser


db = "sqlite+pysqlite:///N:\\Mitarbeiter\\Florian\\projects\\CamTrapPy\\dev.db"
engine = create_engine(db, echo=True, future=True)
Session = sessionmaker(engine, future=True)


def initialize_new_project(p: ProjectParser):
    if database_exists(engine.url):
        drop_database(engine.url)

    Base.metadata.create_all(engine)
    new_project(Session, p)
