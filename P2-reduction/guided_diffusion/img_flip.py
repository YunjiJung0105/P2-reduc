from PIL import Image

for i in range(100):
    img = Image.open('../data/obama/100/{}.jpg'.format(i))
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_flip.save('../data/obama/100/{}.jpg'.format(100+i), 'JPEG')