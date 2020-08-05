import os
from lib import baseclass, logclass

class ProjectPrepare(baseclass.BaseClass):

    folders = [ "config", "dwca", "images", "lists", "log", "models" ]

    def __init__(self):
        super().__init__()

    def make_folders(self):
        for folder in self.folders:
            f = os.path.join(self.project_root, folder)
            if not os.path.exists(f):
                os.mkdir(f)
                os.chmod(f,0o777)
                self.logger.info("created folder \"{}\"".format(f))
            else:
                self.logger.info("folder \"{}\" already exists".format(f))

if __name__ == "__main__":

    prepare = ProjectPrepare()
    prepare.set_project(os.environ)
    prepare.make_folders()
