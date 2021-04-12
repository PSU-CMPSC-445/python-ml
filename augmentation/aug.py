##########################################################################
# This program will take a directory of images and create altered copies #
# One horizontal flip, one vertical flip, and 23 15 degree rotations     #
##########################################################################

from PIL import Image
import glob
import os

# change this directory to the directory of images you want to alter, does not search recursively
ROOT_DIR = 'C:/Users/sandy/OneDrive/Documents/1. PennState/CMPSC445/Group Project/Code Tests/images/2357/'
image_list = []
fn = []

# Create glob for iteration
for filename in glob.glob(ROOT_DIR + '*.png'):
    im = Image.open(filename)
    image_list.append(im)

# Create filename for append
file_list = os.listdir(ROOT_DIR)
for files in file_list:
    fn.append(os.path.splitext(files)[0])

# flip horizontally and vertically
i = 0
for image in image_list:
    hoz_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    hoz_flip.save(ROOT_DIR + 'aug_hozflip_' + fn[i] + '.jpeg', format='JPEG')
    vert_flip = image.transpose(Image.FLIP_TOP_BOTTOM)
    vert_flip.save(ROOT_DIR + 'aug_vertflip_' + fn[i] + '.jpeg', format='JPEG')
    i = i + 1

# rotate in 15 deg increments
i = 0
for image in image_list:
    for x in range(15, 360, 15):
        img_rotate = image.rotate(x)
        img_rotate.save(ROOT_DIR + 'aug_rotate_' + fn[i] + str(x) + '.jpeg', format='JPEG')
    i = i + 1

