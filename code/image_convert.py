import os, csv
import  glob
from lib import baseclass, utils

class ImageConvert(baseclass.BaseClass):

    extensions_to_convert=[ { "extension" : ".png", "converter" : "convert_png" } ]
    files_to_convert=[]

    def __init__(self):
        super().__init__()

    def get_images_to_convert(self):

        extensions = [ x["extension"] for x in self.extensions_to_convert ]

        for f in glob.iglob(self.image_path + '/**/*.*', recursive=True):
            if os.path.isfile(f):
                filename, file_extension = os.path.splitext(f)
                if file_extension.lower() in extensions:
                    self.files_to_convert.append({ "filename" : f, "extension" : file_extension })

    def run_conversions(self):
        for item in self.files_to_convert:
            converter = [ x["converter"] for x in self.extensions_to_convert if x["extension"] == item["extension"] ].pop()
            method_to_call = getattr(self, converter)
            result = method_to_call(item["filename"])

    def convert_png(self,img):
        print("convert_png: {}".format(img))


    def read_downloaded_images_file(self,image_col):

        with open(self.downloaded_images_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=utils._determine_csv_separator(self.downloaded_images_file,"utf-8-sig"))
            for row in csv_reader:
                print(row[image_col])



if __name__ == "__main__":

    ic = ImageConvert()

    ic.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    ic.set_project(os.environ)
    # ic.set_model_folder()
    # ic.set_image_list_file(os.getenv('IMAGE_LIST_FILE'))
    ic.get_images_to_convert()
    ic.run_conversions()
    ic.read_downloaded_images_file(image_col=int(os.environ["IMAGE_LIST_FILE_COLUMN"]) if "IMAGE_LIST_FILE_COLUMN" in os.environ else 2)
