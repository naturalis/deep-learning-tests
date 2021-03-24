import os
import  glob
from lib import baseclass

class ImageConvert(baseclass.BaseClass):

    extensions_to_convert=[ { "extension" : ".png", "converter" : self.fuck() } ]
    files_to_convert=[]

    def __init__(self):
        super().__init__()

    def get_images_to_convert(self):

        extensions = [ x["extension"] for x in self.extensions_to_convert ]

        for f in glob.iglob(self.image_path + '/**/*.*', recursive=True):
            if os.path.isfile(f):
                filename, file_extension = os.path.splitext(f)
                if file_extension.lower() in extensions:
                    # print(f)
                    self.files_to_convert.append({ "filename" : filename, "extension" : file_extension })

        print(self.files_to_convert)

        # bad_files=[]
        # checked=0
        # for filename in glob.iglob(rootdir + '/**/*.jpg', recursive=True):
        #     if os.path.isfile(filename):
        #         # print(filename)
        #         try:
        #             # img = Image.open(filename) # open the image file
        #             # img.verify() # verify that it is, in fact an image
        #             im = Image.open(filename)
        #             im.verify() #I perform also verify, don't know if he sees other types o defects
        #             # im.close() #reload is necessary in my case
        #             im = Image.open(filename)
        #             im.transpose(Image.FLIP_LEFT_RIGHT)
        #             # im.close()
        #         except Exception as e:
        #             bad_files.append({ "filename": filename, "error": str(e)})
        #         checked += 1
        #         if checked % 1000 == 0:
        #             print("{} / {} ".format(len(bad_files),checked))

        # return bad_files



if __name__ == "__main__":

    ic = ImageConvert()

    ic.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    ic.set_project(os.environ)
    # ic.set_model_folder()
    # ic.set_image_list_file(os.getenv('IMAGE_LIST_FILE'))
    ic.get_images_to_convert()
