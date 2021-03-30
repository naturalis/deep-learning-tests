import os, argparse
from lib import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_list",type=str,help="specify downloaded image list",required=True)
    parser.add_argument("--filepath_col",type=int,help="specify image list filepath column",required=True)
    parser.add_argument("--override_image_root_folder",type=str)
    parser.add_argument("--prepend_image_root_folder",type=str)
    args = parser.parse_args()

    dataset = dataset.DataSet()
    dataset.set_environ(os.environ)
    dataset.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    dataset.set_project(os.environ)

    img = utils.ImageVerifier()
    img.set_image_list(args.image_list)
    img.set_image_list_filepath_column(args.filepath_col)
    # replaces whatever the folder is in the image list (just keeps basename)
    if args.override_image_root_folder:
        img.set_override_image_root_folder(args.override_image_root_folder)

    # prepends to whatever is in the image list
    if args.prepend_image_root_folder:
        img.set_prepend_image_root_folder(args.prepend_image_root_folder)

    img.verify_images_from_image_list()
    bad_files = img.get_bad_files()

    for file in bad_files:
        dataset.logger.info("found corrupt file: {} ({})".format(file["filename"],file["error"]))
