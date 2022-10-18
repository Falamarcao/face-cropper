from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import cv2

def add_corners (im, rad):
    im = Image.fromarray(im)
    circle = Image.new('L', (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill = 255)
    alpha = Image.new('L', im.size, 255)
    w, h = im.size
    alpha.paste(circle.crop ((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop ((0, rad, rad, rad * 2)), (0, h-rad))
    alpha.paste(circle.crop ((rad, 0, rad * 2, rad)), (w-rad, 0))
    alpha.paste(circle.crop ((rad, rad, rad * 2, rad * 2)), (w-rad, h-rad))
    im.putalpha(alpha)
    return im

def generate_photo(path, file_name, width=354, height=472):
    padding = (height, width)

    #img = cv2.imread(path + file_name, cv2.IMREAD_COLOR)
       
    stream = open(path + file_name, "rb")
    numpyarray = np.asarray(bytearray(stream.read()), dtype=np.uint8)
    img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 6)

    if len(faces) != 1:
        print('trying profileface 4')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) != 1:
        print('trying profileface 3')
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    if len(faces) != 1:
        print('trying frontalface default 6')
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 6)

    if len(faces) != 1:
        print('trying frontalface default 4')
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    if len(faces) != 1:
        print('trying frontalface default 3')
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    
    if len(faces) != 1:
        print('trying frontalface alt 6')
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        faces = face_cascade.detectMultiScale(gray, 1.2, 6)

    if len(faces) != 1:
        print('trying frontalface alt 4')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) != 1:
        print('trying frontalface alt 3')
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    if len(faces) > 0:
        (x,y,w,h) = faces[0]
        img = img[y-padding[0]:y+h+padding[0]+height, x-padding[1]:x+w+padding[1]]

        img = add_corners(img, 100) #Execute the rounded method with arguments

        img.save(f'output/{file_name[:-4]}-cropped.png')
    else:
        print('No faces recognized')

if __name__ == "__main__":
    from os import listdir

    path = 'photos/'
    file = 'anne.silva.png'

    for file in listdir(path):
        try:
            print(file)
            generate_photo(path, file)
        except:
            next    