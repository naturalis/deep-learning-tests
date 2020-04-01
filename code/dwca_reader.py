import csv
import json
import os
from lib import logclass
from lib import baseclass


class ClassificationClass:
    name = None
    images = []
    ids = []

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def add_images(self, images):
        self.images = self.images + images

    def get_images(self):
        return self.images

    def add_ids(self, ids):
        self.ids = self.ids + ids

    def get_ids(self):
        return self.ids


class DwcaReader(baseclass.BaseClass):
    """
  DwCA reader:
  - extracts an imagelist from the occurence file from a DwCA archive
  """

    project_root = None
    dwca_file_path = None
    dwca_column_headers = {"id": None, "taxon": None, "images": None}
    dwca_column_indices = {"id": None, "taxon": None, "images": None}
    classification_classes = []
    class_image_minimum = 2

    total_images = 0

    def __init__(self):
        self.logger = logclass.LogClass(self.__class__.__name__)

    def set_dwca_file_path(self, dwca_file_path=None):
        if dwca_file_path is not None:
            self.dwca_file_path = dwca_file_path
        else:
            self.dwca_file_path = os.path.join(self.project_root, 'dwca', 'Occurrence.txt')

        if not os.path.isfile(self.dwca_file_path):
            raise FileNotFoundError("DWCA file not found: {}".format(self.dwca_file_path))

    def set_dwca_column_headers(self, id_column=None, taxon_column=None, images_column=None):
        if id_column is not None:
            self.dwca_column_headers["id"] = id_column
        if taxon_column is not None:
            self.dwca_column_headers["taxon"] = taxon_column
        if images_column is not None:
            self.dwca_column_headers["images"] = images_column

    def read_dwca_file(self):
        self._read_dwca_column_headers()
        self._read_dwca_file()

    def _read_dwca_column_headers(self):
        # reading the header line
        with open(self.dwca_file_path, 'r', encoding='utf-8-sig') as file:
            c = csv.reader(file)
            row1 = next(c)

        # finding the column index for each relevant header
        for column in self.dwca_column_headers:
            index = row1.index(self.dwca_column_headers[column])
            self.dwca_column_indices[column] = index

        for column, index in self.dwca_column_indices.items():
            if index is None:
                raise ValueError("column not found in DwCA-file: {}".format(column))

    def _read_dwca_file(self):
        with open(self.dwca_file_path, 'r', encoding='utf-8-sig') as file:
            c = csv.reader(file)
            for row in c:
                class_name = row[self.dwca_column_indices["taxon"]]
                y = [x for x in self.classification_classes if x.get_name() == class_name]
                if len(y) != 0:
                    this_class = y[0]
                else:
                    this_class = ClassificationClass(class_name)
                    self.classification_classes.append(this_class)

                this_class.add_images(row[self.dwca_column_indices["images"]].split("|"))
                this_class.add_ids(row[self.dwca_column_indices["id"]].split("|"))

    def set_class_image_minimum(self, class_image_minimum):
        self.class_image_minimum = class_image_minimum

    def get_class_list(self, apply_class_image_minimum=True):
        return [x for x in self.classification_classes
                if len(x.get_images()) >= self.class_image_minimum
                or self.class_image_minimum == 0]

    def save_lists(self):
        self._save_class_list()
        self._save_image_list()

    def _save_class_list(self):
        tmp = []
        with open(self.class_list_file_csv, 'w') as csvfile:
            c = csv.writer(csvfile, delimiter=',', quotechar='"')
            for item in self.get_class_list():
                c.writerow([item.get_name(), len(item.get_images())])
                tmp.append({"class": item.get_name(), "images": len(item.get_images())})

        with open(self.class_list_file_json, 'w') as outfile:
            json.dump(tmp, outfile)

        self.logger.info(
            "wrote {} classes with min {} images to {}".format(len(self.get_class_list()), self.class_image_minimum,
                                                               self.class_list_file_json))

    def _save_image_list(self):
        with open(self.image_list_file_csv, 'w') as csvfile:
            c = csv.writer(csvfile, delimiter=',', quotechar='"')
            for item in self.get_class_list():
                self.total_images += len(item.get_images())
                for image in item.get_images():
                    c.writerow([item.get_name(), image])

        self.logger.info("wrote {} images to {}".format(self.total_images, self.image_list_file_csv))

    def get_settings(self):
        return {
            "config": {
                "project_root": self.project_root,
                "dwca_file": self.dwca_file_path,
                "dwca_column_headers": self.dwca_column_headers,
                "class_image_minimum": self.class_image_minimum,
            },
            "output": {
                "classes": len(self.get_class_list()),
                "total_images": self.total_images,
                "class_list_files": [self.class_list_file_json, self.class_list_file_csv],
                "image_list_file": self.image_list_file_csv,
            }
        }


if __name__ == "__main__":

    project_root = os.environ['PROJECT_ROOT']

    reader = DwcaReader()
    reader.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    reader.set_project_folders(project_root=project_root)

    if 'DWCA_FILE_PATH' in os.environ:
        reader.set_dwca_file_path(os.environ['DWCA_FILE_PATH'])
    else:
        reader.set_dwca_file_path()

    if 'CLASS_IMAGE_MINIMUM' in os.environ:
        reader.set_class_image_minimum(int(os.environ['CLASS_IMAGE_MINIMUM']))

    reader.set_dwca_column_headers(id_column="catalogNumber", taxon_column="scientificName",
                                   images_column="associatedMedia")
    reader.read_dwca_file()
    reader.save_lists()

    settings = reader.get_settings()
    classes = []
    for item in reader.get_class_list():
        classes.append(
            {"class": item.get_name(), "id_count": len(item.get_ids()), "image_count": len(item.get_images())})

    print(json.dumps({"settings": settings, "classes": classes}))
