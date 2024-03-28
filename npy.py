import numpy as np
from PIL import Image
import cv2
import glob 

# files = glob.glob("/home/ec2-user/tts2/BTCV/data/BTCV/train_npz/*.npz")

# for f in files:
#     x = np.load(f, mmap_mode='r')
#     print(x['label'].min(), x['label'].max(), f)
class_dict = {
        0:(0, 0, 0),
        1:(255, 60, 0),
        2:(255, 60, 32),
        3:(34, 79, 117),
        5:(117, 200, 91),
        6:(230, 91, 101),
        7:(255, 0, 155),
        8:(75, 105, 175)
}

def onehot_to_rgb(onehot, color_dict=class_dict):
    onehot = np.int64(onehot)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[onehot==k] = color_dict[k]
    return np.uint8(output)

x =  np.load("/home/ec2-user/tts2/BTCV/data/BTCV/train_npz/case0010_slice142.npz")
print(x['image'].shape)

im1 = Image.fromarray(np.uint8(x['image']*255)).convert('RGB')


im2 = Image.fromarray(onehot_to_rgb(x['label']))


im1.save('test.png')
im2.save('test2.png')


