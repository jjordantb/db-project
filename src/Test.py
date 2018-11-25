from PIL import Image, ImageDraw, ImageOps
import PIL
from tkinter import *

width = 128
height = 128
center = height//2
white = (255, 255, 255)

root = Tk()


def save():
    im = ImageOps.invert(image1)
    im.thumbnail((28, 28))
    pixels = list(im.getdata())
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    raw = []
    for i in pixels:
        for j in i:
            raw.append(j[0])
    root.destroy()
    return raw


def paint(event):
    x1, y1 = (event.x - 2), (event.y - 2)
    x2, y2 = (event.x + 2), (event.y + 2)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=5)
    draw.line([x1, y1, x2, y2], fill="black", width=5)


cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)
cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)
button = Button(text="query", command=save)
button.pack()
root.mainloop()
