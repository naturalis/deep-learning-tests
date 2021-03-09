import os, argparse, requests, json, csv
from urllib.parse import urlparse

class BatchApiCall:

    input_dir = None
    image_list = None
    api_url = None
    download_folder = None
    images = []

    def __init__(self):
        pass

    def __del__(self):
        pass

    def set_input_dir(self,input_dir):
        self.input_dir = input_dir

    def set_image_list(self,image_list):
        self.image_list = image_list

    def set_api_url(self,api_url):
        self.api_url = api_url

    def set_download_folder(self,download_folder):
        self.download_folder = download_folder

    def get_images(self):
        if not self.input_dir is None:
            self.get_images_from_folder()
        elif not self.image_list is None:
            self.get_images_from_list()

    def get_images_from_folder(self):
        if not os.path.exists(self.input_dir):
            raise ValueError("path doesn't exist: {}".format(self.input_dir))

        for subdir, dirs, files in os.walk(self.input_dir):
            for file in files:
                this_class = subdir.replace(self.input_dir,'')
                filepath = subdir + os.sep + file
                if filepath.endswith(".jpg"):
                    self.images.append({'class':this_class,'file':filepath})

        print("found {} images".format(len(self.images)))

    def get_images_from_list(self):
        if not os.path.exists(self.image_list):
            raise ValueError("file doesn't exist: {}".format(self.image_list))

        with open(self.image_list, 'r', encoding='utf-8-sig') as file:
            c = csv.reader(file)
            for row in c:

                if not row:
                    continue

                if not len(row)>=2:
                    continue

                url = urlparse(row[1])

                if url.scheme and url.netloc:
                    # print("url")
                    self.images.append({'class':row[0],'url':row[1]})
                else:
                    # print("file")
                    if row[1].endswith(".jpg"):
                        self.images.append({'class':row[0],'file':row[1]})

        print("found {} images".format(len(self.images)))

    def download_images(self):
        if not self.download_folder:
            raise ValueError("no download folder (list contains URLs)")

        if not os.path.exists(self.download_folder):
            raise ValueError("download folder {} doesn't exist".format(self.download_folder))




    def run_identifications(self):

        print("image\tclass\tprediction")

        for image in self.images:
            try:
                this_class = os.path.dirname(image.replace(self.input_dir,''))
                this_file = os.path.basename(image.replace(self.input_dir,''))
                with open(image, "rb") as file:
                    myobj = {'image':  file }
                    response = requests.post(self.api_url, files=myobj)
                    p = json.loads(response.text)
                    print("{}\t[{}]\t{}\t[{}]\t{}".format(
                        "V" if this_class==p["predictions"][0]["class"] else "-",
                        this_class,
                        this_file,
                        p["predictions"][0]["class"],
                        p["predictions"][0]["prediction"])
                    )
            except Exception as e:
                print("{}: {}".format(image, e))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_list",type=str)
    parser.add_argument("--image_folder",type=str)
    parser.add_argument("--download_folder",type=str)
    parser.add_argument("--api_url",type=str, required=True)
    args = parser.parse_args()

    if not args.image_folder and not args.image_list:
        parser.print_help()
        raise ValueError("need image list or image root folder")

    if args.image_folder and args.image_list:
        parser.print_help()
        raise ValueError("need image list or image root folder")

    bac = BatchApiCall()

    if args.image_list:
        bac.set_image_list(args.image_list)

    if args.image_folder:
        bac.set_input_dir(args.image_folder)

    if args.api_url:
        bac.set_api_url(args.api_url)

    if args.download_folder:
        bac.set_download_folder(args.download_folder)

    bac.get_images()
    bac.download_images()
    # bac.run_identifications()
