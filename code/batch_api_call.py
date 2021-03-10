import os, argparse, requests, json, csv

class BatchApiCall:

    input_dir = None
    image_list = None
    api_url = None
    override_image_root_folder = None
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

    def set_override_image_root_folder(self,folder):
        self.override_image_root_folder = folder

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

                if row[1].endswith(".jpg"):
                    file = row[1]
                elif row[2].endswith(".jpg"):
                    file = row[2]

                if self.override_image_root_folder:
                    file = os.path.join(self.override_image_root_folder, os.path.basename(file))

                if row[0] and file:
                    self.images.append({'class':row[0],'file':file})

        print("found {} images".format(len(self.images)))


    def run_identifications(self):

        print("in_model,match,top_3,class,predicted_class,prediction")

        for image in self.images:
            try:
                this_class = image['class']
                this_file = image['file']
                with open(image['file'], "rb") as file:
                    myobj = {'image':  file }
                    response = requests.post(self.api_url, files=myobj)
                    p = json.loads(response.text)

                    result_class = p["predictions"][0]["class"]
                    result_prediction = p["predictions"][0]["prediction"]
                    is_match = this_class==p["predictions"][0]["class"]
                    in_top_3 = this_class in [p["predictions"][0]["class"],p["predictions"][1]["class"],p["predictions"][2]["class"]]

                    in_model = '-'
                    for i,item in enumerate(p["predictions"]):
                        if item["class"]==this_class:
                            in_model = i
                            break


                    print("{},{},{},{},{},{}".format(
                        in_model,
                        'v' if is_match else '-',
                        'v' if in_top_3 else '-',
                        this_class,
                        p["predictions"][0]["class"],
                        p["predictions"][0]["prediction"])
                    )
            except Exception as e:
                print("{}: {}".format(image, e))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_list",type=str)
    parser.add_argument("--image_folder",type=str)
    parser.add_argument("--api_url",type=str, required=True)
    parser.add_argument("--override_image_root_folder",type=str)
    args = parser.parse_args()

    if not args.image_folder and not args.image_list:
        parser.print_help()
        raise ValueError("image list or image root folder required")

    if args.image_folder and args.image_list:
        parser.print_help()
        raise ValueError("need either image list or image root folder")

    bac = BatchApiCall()

    if args.image_list:
        bac.set_image_list(args.image_list)

    if args.image_folder:
        bac.set_input_dir(args.image_folder)

    if args.api_url:
        bac.set_api_url(args.api_url)

    if args.override_image_root_folder:
        bac.set_override_image_root_folder(args.override_image_root_folder)

    bac.get_images()
    bac.run_identifications()

    #  are they in the model at all