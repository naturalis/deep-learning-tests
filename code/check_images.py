import os, argparse
from lib import baseclass, utils, dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root",type=str)
    args = parser.parse_args()

    dataset = dataset.DataSet()
    dataset.set_environ(os.environ)
    dataset.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    dataset.set_project(os.environ)
    dataset.logger.info("checking for corrupt files")

    img = utils.ImageVerifier()

    if args.image_root:
        dataset.logger.info("checking for corrupt files in {}".format(args.image_root))
        img.set_root_dir(args.image_root)
    else:
        dataset.logger.info("checking for corrupt files in {}".format(dataset.image_path))
        img.set_root_dir(dataset.image_path)

    img.verify_images_from_folder()
    bad_files = img.get_bad_files()

    for file in bad_files:
        dataset.logger.info("found corrupt file: {} ({})".format(file["filename"],file["error"]))
