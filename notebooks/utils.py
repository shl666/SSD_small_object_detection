import cv2
import re
import matplotlib.image as mpimg

def crop_img(path, path_cropped, image_name, nums):
    im = cv2.imread(path + image_name)
    imgheight, imgwidth, _ = im.shape
    path_save_lst = []
    for num in nums:
        crop_width = imgwidth//num
        crop_height = imgheight//num
        for i in range(num):
            for j in range(num):
                box = (i*crop_width, j*crop_height, (i+1)*crop_width, (j+1)*crop_height)
                img = im[box[1]:box[3], box[0]:box[2]]
                image_name_save = '{}_{}_({},{}).{}'.format(image_name.split('.')[0],num,i,j,image_name.split('.')[1])
                path_save = path_cropped+image_name_save
                cv2.imwrite(path_save, img)
                path_save_lst.append(path_save)
    return path_save_lst
