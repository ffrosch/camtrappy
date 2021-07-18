from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, drop_database

from camtrappy.db.schema import Base

engine = create_engine("sqlite+pysqlite:///N:\\Mitarbeiter\\Florian\\projects\\CamTrapPy\\dev.db", echo=True, future=True)

if database_exists(engine.url):
    drop_database(engine.url)
Base.metadata.create_all(engine)

Session = sessionmaker(engine, future=True)
