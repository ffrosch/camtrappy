from camtrappy.db.schema import Project, Location, Video
from camtrappy.io.parsing import ProjectParser


def new_project(Session, p: ProjectParser):
    with Session.begin() as session:
        # bulk insert is used, even for single items,
        # to enforce resolution order
        session.bulk_insert_mappings(Project,
                                     [dict(name=p.name,
                                           projectfolder=p.projectfolder,
                                           datafolder=p.datafolder)])

        project = session.query(Project).filter(Project.name == p.name).scalar()
        for name in p.locations:
            session.bulk_insert_mappings(Location,
                                         [dict(name=name,
                                               folder=name,
                                               project_id=project.id)])

            location_id = (session.query(Location)
                           .filter(Location.name == name)
                           .scalar()).id
            data = p.data(name)
            for video in data:
                video['location_id'] = location_id
            session.bulk_insert_mappings(Video, data)