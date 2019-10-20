from PIL import Image
import numpy as np

def open_img(filename):
    img = Image.open(filename).convert('RGB')
    return np.array(img, dtype=np.uint8)

def concatenate_photos(a, b, name):
    x = open_img(a)
    y = open_img(b)
    z = np.concatenate((x, np.zeros((100, x.shape[1], 3), dtype=np.uint8), y))
    img = Image.fromarray(z)
    img.save('sim/' + name + '.png')

def get_name(month, keyword):
    return 'sim/' + 'big_' + keyword + '2017' + str(month) + '.png' 

if __name__ == "__main__":
    for i in range(1, 13):
        a = get_name(i, 'pred')
        b = get_name(i, 'real')
        concatenate_photos(a, b, 'con'+str(i))

#convert -delay 20 -loop 0 con*.png res.gif