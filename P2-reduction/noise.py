import numpy as np
from PIL import Image

a = np.random.rand(120000)*255
a = a.reshape((200,200,3))
a = a.astype(np.uint8)
a = Image.fromarray(a) 
a.save('samples/1d.jpg', 'JPEG')

b = np.random.rand(200,200,3)*255
b = b.astype(np.uint8)
b = Image.fromarray(b)
b.save('samples/3d.jpg', 'JPEG')
