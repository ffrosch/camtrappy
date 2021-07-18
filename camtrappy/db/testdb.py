from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, drop_database

from camtrappy.db.schema import Base, Project, Location, Video

engine = create_engine("sqlite+pysqlite:///N:\\Mitarbeiter\\Florian\\projects\\CamTrapPy\\dev.db", echo=True, future=True)

if database_exists(engine.url):
    drop_database(engine.url)
Base.metadata.create_all(engine)

Session = sessionmaker(engine, future=True)


def initialize_new_project(DATA_PROJECT):
    with Session.begin() as session:
        session.bulk_insert_mappings(Project, [dict(name=DATA_PROJECT.name,
                                                    projectfolder=DATA_PROJECT.projectfolder,
                                                    datafolder=DATA_PROJECT.datafolder)])
        project = session.query(Project).filter(Project.name == DATA_PROJECT.name).scalar()

        for name, data in DATA_PROJECT._data.items():
            session.bulk_insert_mappings(Location, [dict(name=name, project_id=project.id)])
            location = session.query(Location).filter(Location.name == name).scalar()

            session.bulk_insert_mappings(Video, [
                                            dict(path=path, date=date, starttime=time, location_id=location.id)
                                            for path, date, time in zip(data['videos'], data['dates'], data['times'])
                                        ])

