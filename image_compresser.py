from PIL import Image
import PIL
import os

# Compresses collected images

DATA_FOLDER_PATH = './data/'
COMPRESSED_FOLDER_PATH = './data/compressed/'


# Rescales images and saves it
def rescale_image_and_save(file_name, original_path=DATA_FOLDER_PATH, compression_path=COMPRESSED_FOLDER_PATH):
    image = Image.open(original_path + file_name)
    image.thumbnail((256, 256), PIL.Image.ANTIALIAS)
    image.save(compression_path + file_name)


# Do not compress images that are already compressed
def iterate_over_data_folder():
    for filename in os.listdir(DATA_FOLDER_PATH):
        if filename.endswith('.jpg'):
            if os.path.isfile(COMPRESSED_FOLDER_PATH + filename):
                continue
            print(filename)
            rescale_image_and_save(filename)
        else:
            continue


if __name__ == '__main__':
    iterate_over_data_folder()
