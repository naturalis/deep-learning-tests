import json
import os, csv, re
from urllib.parse import urlparse
from hashlib import md5
import urllib.request
from lib import logclass

class ImageDownloader():
    project_root = None
    download_root_folder = None
    image_list_file = None
    image_list = []
    previously_downloaded_files = []
    skip_download_if_exists = True
    image_default_extension = None
    image_url_to_name = None
    downloaded_images_file = None
    downloaded_images_filename = 'lists/downloaded_images.csv'
    subfolder_max_files = 1000
    _subfolder_index = 0
    _current_subdir = None

    def __init__(self):
        self.logger = logclass.LogClass(self.__class__.__name__)

    def set_project_root(self, project_root):
        self.project_root = project_root

        if not os.path.isdir(self.project_root):
            raise FileNotFoundError("project root doesn't exist: {}".format(self.project_root))

        self.download_root_folder = 'images'

        if not os.path.isdir(os.path.join(self.project_root, self.download_root_folder)):
            os.mkdir(os.path.join(self.project_root, self.download_root_folder))

        self.downloaded_images_file = os.path.join(self.project_root, self.downloaded_images_filename)

    def set_image_url_to_name(self, image_url_to_name):
        self.image_url_to_name = image_url_to_name

    def set_image_list_file(self, image_list_file=None):
        if image_list_file is not None:
            self.image_list_file = image_list_file
        else:
            self.image_list_file = os.path.join(self.project_root, 'lists', 'images.csv')

        if not os.path.isfile(self.image_list_file):
            raise FileNotFoundError("image list file  not found: {}".format(self.image_list_file))

    def set_subfolder_max_files(self, max_files):
        if isinstance(max_files, int) and 100 < int < 100000:
            self.subfolder_max_files = max_files
        else:
            raise ValueError("max subfolder files must be int between 100 and 100000 ({})".format(max_files))

    def set_skip_download_if_exists(self, state):
        if isinstance(state, bool):
            self.skip_download_if_exists = state
        else:
            raise ValueError("skip download if exists must be a boolean ({})".format(state))

    def read_image_list(self):
        with open(self.image_list_file, 'r', encoding='utf-8-sig') as file:
            c = csv.reader(file)
            for row in c:
                self.image_list.append(row)

    def _get_previously_downloaded_files(self):
        for subdir, dirs, files in os.walk(os.path.join(self.project_root, self.download_root_folder)):
            for file in files:
                self.previously_downloaded_files.append({"file": file, "path": os.path.join(subdir.replace(os.path.join(self.project_root, self.download_root_folder),""),file).lstrip("/")})

    def download_images(self):
        if self.skip_download_if_exists:
            self._get_previously_downloaded_files()

        downloaded = 0
        failed = 0
        skipped = 0

        with open(self.downloaded_images_file, 'w') as csvfile:
            c = csv.writer(csvfile, delimiter=',', quotechar='"')

            for item in self.image_list:
                url = item[1]
                p = urlparse(url)

                self._set_download_subdir()

                if self.image_url_to_name is not None:
                    filename = re.sub(self.image_url_to_name['expression'], self.image_url_to_name['replace'], p.path)
                    filename += self.image_url_to_name['postfix']
                else:
                    filename = os.path.basename(p.path)

                if self.skip_download_if_exists:
                    existing_images = [x for x in self.previously_downloaded_files if x["file"] == filename]
                    skip_download = len(existing_images) > 0
                else:
                    skip_download = False

                if skip_download:
                    c.writerow([item[0], url, existing_images[0]["path"]])
                    self.logger.info("skipped downloading {}".format(url))
                    skipped += 1
                else:
                    file_to_save = os.path.join(self.project_root, self.download_root_folder, self._current_subdir,
                                                filename)
                    try:
                        urllib.request.urlretrieve(url, file_to_save)
                        c.writerow([item[0], url, os.path.join(self._current_subdir, filename)])
                        self.logger.info("downloaded {} to {} ".format(url, file_to_save))
                        downloaded += 1
                    except Exception as e:
                        self.logger.error("could not download {}: {}".format(url, e))
                        failed += 1

        self.logger.info("downloaded {}, skipped {}, failed {}".format(downloaded, skipped, failed))

    def _set_download_subdir(self):
        self._current_subdir = md5(str(self._subfolder_index).encode('utf-8')).hexdigest()[:10]
        subdir_path = os.path.join(self.project_root, self.download_root_folder, self._current_subdir)
        if not os.path.isdir(subdir_path):
            os.mkdir(subdir_path)

        n = len([name for name in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, name))])
        if n >= self.subfolder_max_files:
            while True:
                self._subfolder_index += 1
                self._current_subdir = md5(str(self._subfolder_index).encode('utf-8')).hexdigest()[:10]
                if not os.path.isdir(os.path.join(self.project_root, self.download_root_folder, self._current_subdir)):
                    os.mkdir(os.path.join(self.project_root, self.download_root_folder, self._current_subdir))
                    break


if __name__ == "__main__":
    downloader = ImageDownloader()
    downloader.set_project_root(os.environ['PROJECT_ROOT'])

    if 'IMAGE_URL_TO_NAME' in os.environ:
        downloader.set_image_url_to_name(json.loads(os.environ['IMAGE_URL_TO_NAME']))

    if 'IMAGE_LIST_FILE' in os.environ:
        downloader.set_image_list_file(os.environ['IMAGE_LIST_FILE'])
    else:
        downloader.set_image_list_file()

    downloader.read_image_list()
    downloader.download_images()
