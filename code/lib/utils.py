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


def verify_images(rootdir):
    # bad_files = verify_images("/data")
    bad_files=[]
    checked=0
    for filename in glob.iglob(rootdir + '/**/*.jpg', recursive=True):
        if os.path.isfile(filename):
            # print(filename)

            im = Image.open(filename)
            im.verify() #I perform also verify, don't know if he sees other types o defects
            # im.close() #reload is necessary in my case
            im = Image.open(filename) 
            im.transpose(Image.FLIP_LEFT_RIGHT)
            # im.close()

            try:
                # img = Image.open(filename) # open the image file
                # img.verify() # verify that it is, in fact an image
                im = Image.open(filename)
                im.verify() #I perform also verify, don't know if he sees other types o defects
                im.close() #reload is necessary in my case
                im = Image.open(filename) 
                im.transpose(Image.FLIP_LEFT_RIGHT)
                im.close()
            except Exception as e:
                bad_files.append({ "filename": filename, "error": str(e)})
            checked += 1
            if checked % 1000 == 0:
                print("{} / {} ".format(len(bad_files),checked))

    return bad_files
