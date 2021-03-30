import time
import os
from PIL import Image
import  glob

class Timer:

    start_time = None
    end_time = None
    formats = {
        "pretty" : "%02dd %02dh %02dm %02ds"
    }
    milestones = []

    def __init__(self):
        self.set_start_time()

    def set_start_time(self):
        self.start_time = self.get_timestamp()

    def set_end_time(self):
        self.end_time = self.get_timestamp()

    def get_timestamp(self):
        return time.time()

    def add_milestone(self,label):
        self.milestones.append({ "label" : label, "timestamp" : self.get_timestamp() })

    def get_milestones(self):
        return self.milestones

    def reset_milestones(self):
        self.milestones = []

    def get_time_passed(self,format="pretty"):
        if None is self.end_time:
            self.set_end_time()
        time = float(self.end_time - self.start_time)
        day = time // (24 * 3600)
        time = time % (24 * 3600)
        hour = time // 3600
        time %= 3600
        minutes = time // 60
        time %= 60
        seconds = time
        return self.formats[format] % (day, hour, minutes, seconds)


class ImageVerifier:

    root_dir = None
    bad_files = []
    image_list = []
    current_image_file = None
    image_list_file = None
    filepath_col = None
    override_image_root_folder = None
    prepend_image_root_folder = None
    progress = 0

    def set_root_dir(self,root_dir):
        self.root_dir = root_dir

    def set_current_image_file(self,current_image_file):
        self.current_image_file = current_image_file

    def set_image_list_file(self,image_list):
        self.image_list_file = image_list_file

    def set_image_list_filepath_column(self,filepath_col):
        self.filepath_col = filepath_col

    def set_override_image_root_folder(self,folder):
        self.override_image_root_folder = folder

    def set_prepend_image_root_folder(self,folder):
        self.prepend_image_root_folder = folder

    def get_bad_files(self):
        return self.bad_files

    def verify_image(self):
        # img = Image.open(filename) # open the image file
        # img.verify() # verify that it is, in fact an image
        im = Image.open(self.current_image_file)
        im.verify() #I perform also verify, don't know if he sees other types o defects
        # im.close() #reload is necessary in my case
        im = Image.open(self.current_image_file)
        im.transpose(Image.FLIP_LEFT_RIGHT)
        # im.close()

    def verify_images_from_folder(self):
        self.progress = 0
        for filename in glob.iglob(self.root_dir + '/**/*.jpg', recursive=True):
            if os.path.isfile(filename):
                # print(filename)
                try:
                    self.set_current_image_file(filename)
                    self.verify_image()
                except Exception as e:
                    self.bad_files.append({ "filename": filename, "error": str(e)})
                self.progress += 1
                self._print_progress()

    def verify_images_from_image_list(self):
        self.progress = 0
        self.image_list = []
        self.read_image_list_file()
        for filename in self.image_list:
            try:
                self.set_current_image_file(filename)
                self.verify_image()
            except Exception as e:
                self.bad_files.append({ "filename": filename, "error": str(e)})
            self.progress += 1
            self._print_progress()

    def _print_progress(self):
        if self.progress % 1000 == 0:
            print("{} / {} ".format(len(self.bad_files),self.progress))

    def read_image_list_file(self):
        with open(self.image_list_file) as csv_file:
            # reader = csv.reader(csv_file, delimiter=utils._determine_csv_separator(self.downloaded_images_file,"utf-8-sig"))
            reader = csv.reader(csv_file, delimiter=_determine_csv_separator(self.image_list,"utf-8-sig"))
            for row in reader:
                if not row or self.filepath_col not in row:
                    continue
                file = row[self.filepath_col]

                if self.override_image_root_folder:
                    file = os.path.join(self.override_image_root_folder, os.path.basename(file))

                if self.prepend_image_root_folder:
                    file = os.path.join(self.prepend_image_root_folder, file)

                if row[0] and file:
                    self.image_list.append(file)

        print("read image list {}, found {} images".format(self.image_list,len(self.image_list)))



def _determine_csv_separator(filepath,encoding):
    f = open(filepath, "r", encoding=encoding)
    line = f.readline()
    if line.count('\t') > 0:
        sep = '\t'
    else:
        sep = ','
    return sep
