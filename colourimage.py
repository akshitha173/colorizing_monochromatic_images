import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv

# Load the neural network
caffe_net = cv.dnn.readNetFromCaffe("colorization_deploy_v2.prototxt", "colorization_release_v2.caffemodel")
pts_in_hull = np.load('pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
caffe_net.getLayer(caffe_net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
caffe_net.getLayer(caffe_net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

class Window(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Image Colorization")
        self.pack(fill=BOTH, expand=1)

        # Create a canvas to display images side by side
        self.canvas = Canvas(self, width=960, height=360)  # 480x360 for each image
        self.canvas.pack()

        # Add a menu to upload the image
        self.menu = Menu(self.master)
        self.master.config(menu=self.menu)
        file_menu = Menu(self.menu)
        file_menu.add_command(label="UPLOAD IMAGE", command=self.open_image)
        self.menu.add_cascade(label="File", menu=file_menu)

        self.original_image = None  # Placeholder for original grayscale image
        self.colorized_image = None  # Placeholder for colorized image

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Open the image in grayscale
            pil_image = Image.open(file_path).convert('L').resize((480, 360))
            self.original_image = ImageTk.PhotoImage(pil_image)

            # Display the original grayscale image on the left side of the canvas
            self.canvas.create_image(0, 0, image=self.original_image, anchor=NW)

            # Process the image to get the colorized version
            color_image = self.process_and_color(pil_image)

            # Convert the colorized image to an ImageTk object for Tkinter display
            self.colorized_image = ImageTk.PhotoImage(color_image)

            # Display the colorized image on the right side of the canvas
            self.canvas.create_image(480, 0, image=self.colorized_image, anchor=NW)

    def process_and_color(self, pil_image):
        # Convert PIL Image (grayscale) to OpenCV format and replicate channels to make it 3-channel (BGR)
        open_cv_image = np.array(pil_image.convert('RGB'))[:, :, ::-1]  # Convert RGB to BGR for OpenCV

        # Preprocess the image for the Caffe model
        scaled = open_cv_image.astype('float32') / 255.0  # Scale to [0,1]
        lab_image = cv.cvtColor(scaled, cv.COLOR_BGR2Lab)  # Convert to Lab color space
        l_channel = lab_image[:, :, 0]  # Extract the L channel

        # Resize the lightness channel to network input size
        resized_l_channel = cv.resize(l_channel, (224, 224))
        resized_l_channel -= 50  # Subtract the mean value

        # Prepare the model input
        net_input = cv.dnn.blobFromImage(resized_l_channel)

        # Perform inference
        caffe_net.setInput(net_input)
        ab_channels = caffe_net.forward()[0, :, :, :].transpose((1, 2, 0))  # Predict A and B channels

        # Resize the ab_channels to original image size
        ab_channels = cv.resize(ab_channels, (open_cv_image.shape[1], open_cv_image.shape[0]))

        # Merge the L channel with ab_channels
        colorized_image = np.concatenate((l_channel[:, :, np.newaxis], ab_channels), axis=2)
        colorized_image = cv.cvtColor(colorized_image, cv.COLOR_Lab2BGR)
        colorized_image = np.clip(colorized_image * 255, 0, 255).astype('uint8')

        # Convert back to PIL format to display in Tkinter
        return Image.fromarray(cv.cvtColor(colorized_image, cv.COLOR_BGR2RGB))

root = tk.Tk()
root.geometry("960x360")  # Set window size large enough for both images
app = Window(root)
root.mainloop()
