import os, argparse, requests, json

class BatchApiCall:

    input_dir = None
    api_url = None
    images = []

    def __init__(self):
        pass

    def __del__(self):
        pass

    def set_input_dir(self,input_dir):
        self.input_dir = input_dir

    def set_api_url(self,api_url):
        self.api_url = api_url

    def get_images(self):
        if not os.path.exists(self.input_dir):
            raise ValueError("path doesn't exist: {}".format(self.input_dir))

        for subdir, dirs, files in os.walk(self.input_dir):
            for file in files:
                #print os.path.join(subdir, file)
                filepath = subdir + os.sep + file
                if filepath.endswith(".jpg") or filepath.endswith(".png"):
                    self.images.append(filepath)

        print("found {} images".format(len(self.images)))

    def run_identifications(self):

        print("image\tclass\tprediction")

        for image in self.images:
            this_class = image.replace(self.input_dir,'')
            with open(image, "rb") as file:
                myobj = {'image':  file }
                response = requests.post(self.api_url, files=myobj)
                p = json.loads(response.text)
                print("{}\t{}\t{}\t{}".format(
                    "!" if this_class==p["predictions"][0]["class"] else "-"
                    this_class,
                    p["predictions"][0]["class"],
                    p["predictions"][0]["prediction"])
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",type=str, required=True)
    parser.add_argument("--api_url",type=str, required=True)
    args = parser.parse_args()

    bac = BatchApiCall()

    if args.dir:
        bac.set_input_dir(args.dir)

    if args.api_url:
        bac.set_api_url(args.api_url)

    bac.get_images()
    bac.run_identifications()
