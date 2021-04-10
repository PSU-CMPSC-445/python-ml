from PIL import Image
import glob

ROOT_DIR = 'C:/Users/sandy/OneDrive/Documents/1. PennState/CMPSC445/Group Project/Code Tests/images/'

image_list = []
fn = []
for filename in glob.glob(ROOT_DIR + '/2357/*.png'):
    im = Image.open(filename)
    image_list.append(im)
    fn.append(filename)

for image in image_list:

    # flip horizontally and vertically
    # hoz_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    # vert_flip = image.transpose(Image.FLIP_TOP_BOTTOM)

    # rotate
    for x in range(15, 360, 15):
        img_rotate = image.rotate(x)
        img_rotate.save(ROOT_DIR + '/2357/aug_2357 brick corner 1x2x2 000L-' + str(x) + '-.jpeg', format='JPEG')
