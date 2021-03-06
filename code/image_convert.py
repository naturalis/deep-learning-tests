import os, csv, argparse, glob
from shutil import copyfile
from datetime import datetime
from PIL import Image
from lib import baseclass, utils

class ImageConvert(baseclass.BaseClass):

    downloaded_images=[]
    extensions_to_convert=[ { "extension" : ".png", "converter" : "convert_png" } ]
    files_to_convert=[]
    udpdated = 0

    def __init__(self):
        super().__init__()
        self.logger.info("available conversions: {} -> jpg".format(";".join([ x["extension"] for x in self.extensions_to_convert ])))

    def set_image_col(self,image_col):
        self.image_col = image_col

    def read_downloaded_images_file(self):
        with open(self.downloaded_images_file) as csv_file:
            reader = csv.reader(csv_file, delimiter=utils._determine_csv_separator(self.downloaded_images_file,"utf-8-sig"))
            self.downloaded_images = list(reader)

        self.logger.info("read image list {}".format(self.downloaded_images_file))

    def get_images_to_convert(self):
        extensions = [ x["extension"] for x in self.extensions_to_convert ]

        for item in self.downloaded_images:
            f = item[self.image_col]
            filename, file_extension = os.path.splitext(f)
            if file_extension.lower() in extensions:
                self.files_to_convert.append({ "filename" : f, "extension" : file_extension })

        self.logger.info("found {} images to convert".format(len(self.files_to_convert)))

    def run_conversions(self):
        for item in self.files_to_convert:
            converter = [ x["converter"] for x in self.extensions_to_convert if x["extension"] == item["extension"] ].pop()
            method_to_call = getattr(self, converter)
            result = method_to_call(item["filename"])

    def convert_png(self,img):
        new_img = img + '.jpg'
        im = Image.open(os.path.join(self.image_root_path,img))
        rgb_im = im.convert('RGB')
        rgb_im.save(os.path.join(self.image_root_path,new_img))

        for idx, item in enumerate(self.downloaded_images):
            if item[self.image_col] == img:
                self.downloaded_images[idx][self.image_col] = new_img
                self.udpdated += 1

        self.logger.info("converted png: {} --> {}".format(img,new_img))

    def save_updated_image_list(self):
        if self.udpdated > 0:
            filename, file_extension = os.path.splitext(os.path.basename(self.downloaded_images_file))
            backup = os.path.join(os.path.dirname(self.downloaded_images_file), "".join([filename,
                                "--pre-image-convert-", self.get_formatted_timestamp(), file_extension]))

            copyfile(self.downloaded_images_file,backup)
            self.logger.info("backed up original image list: {}".format(backup))

            with open(self.downloaded_images_file, "w") as f:
                writer = csv.writer(f)
                writer.writerows(self.downloaded_images)

            self.logger.info("wrote updated image list")


if __name__ == "__main__":

    ic = ImageConvert()

    parser = argparse.ArgumentParser()
    parser.add_argument("--alt_image_list",type=str,help="specify alternative downloaded image list")
    args = parser.parse_args()

    ic.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    ic.set_project(os.environ)
    ic.set_image_col(int(os.environ["IMAGE_LIST_FILE_COLUMN"]) if "IMAGE_LIST_FILE_COLUMN" in os.environ else 2)

    if args.alt_image_list:
        ic.set_alt_downloaded_images_file(args.alt_image_list)

    ic.read_downloaded_images_file()
    ic.get_images_to_convert()
    ic.run_conversions()
    ic.save_updated_image_list()
