import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import *
from PIL import ImageTk, Image, ImageEnhance
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os
import sys
import cv2
import numpy as np
import colorsys
import matplotlib.pyplot as plt
import skimage.io as data
from tkinter.ttk import Progressbar
from itertools import combinations
from tkinter.filedialog import askopenfilenames, askdirectory
from tkinter.simpledialog import askinteger, askfloat, askstring
from random import random


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Configure the root window
        self.iconbitmap('assets/img/logo.ico')
        self.title('OK Image')
        self.state('zoomed')
        self.resizable(width=True, height=True)                

        # Data Menu
        self.menubar = Menu(self)
        self.filemenu_data = Menu(self.menubar, tearoff=0)
        self.filemenu_data.add_command(label="Software", command=self.about)
        self.filemenu_data.add_command(label="Developer", command=self.dev)

        # Menu Names
        self.menubar.add_cascade(label="About", menu=self.filemenu_data)

    # About System
    def about(self):
        tk.messagebox.showinfo("About", "OK Image\nversion 1.0")

    # Developer
    def dev(self):
        tk.messagebox.showinfo("Developer", "Igor Machado de Castro\nwww.linkedin.com/in/machado-de-castro")


    # Browse directory
    def openFile(self):
        global preview, imageList
        image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
        file_path_list = askopenfilenames(filetypes=image_formats, initialdir="/", title='Please select a picture to analyze')        
        imageList = []
        n = 1        
        listBox.delete(0,'end') # clear listbox

        # start progress bar
        pb = Progressbar(
            self,
            orient = HORIZONTAL,
            length = len(file_path_list),
            mode = 'determinate',
            maximum=len(file_path_list)
            )
        pb.pack(side="bottom", fill="x")        
        pb.start()

        for i in file_path_list:
            image = data.imread(i)
            scale_percent = 50 # percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            preview = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)            
            imageList.append(i)  
            # insert into listbox         
            listBox.insert("end", i)
            n += 1   

            # progress bar
            pb["value"] = n+1
            self.update()            
        pb.stop() 
        pb.destroy()  

                        
    # Open image
    def openImage(self): 
        global oki, label          
        fn = self.openFile()
        oki = Image.fromarray(fn)        
        # Load the image file.
        img = ImageTk.PhotoImage(image=oki)    
        # Create the canvas, size in pixels.        
        label = tk.Label(self, image=img)  
        label.image = img 
        label.pack()  
              

    # Clear listbox
    def clear(self):        
        for i in imageList:
            imageList.clear()
        listBox.delete(0,'end')

    # Circle Mask Custom Preview
    def circleMaskCustom(self):
        radius = askinteger("Circle Radius", "Value:", parent=self)
        cX = askinteger("Rectangle X Position", "Value:", parent=self)
        cY = askinteger("Rectangle Y Position", "Value:", parent=self)
        n = 0
        for i in imageList:
            image = cv2.imread(i)            
            mask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.circle(mask, (cX, cY), radius, 255, -1)
            masked = cv2.bitwise_and(image, image, mask = mask)
            fname = os.path.split(i)[1]
            fextension = os.path.split(i)[-1]
            cv2.imwrite(f'./circle_mask/{fname}_{n}_h.{fextension}', masked)
            n += 1 
        imageList.clear()
        listBox.delete(0,'end') # clear listbox

    # Circle Mask Centralized Preview
    def circleMaskCentralized(self):
        radius = askinteger("Circle Radius", "Value:", parent=self)
        n = 0
        for i in imageList:
            image = cv2.imread(i)            
            (cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
            mask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.circle(mask, (cX, cY), radius, 255, -1)
            masked = cv2.bitwise_and(image, image, mask = mask)
            fname = os.path.split(i)[1]
            fextension = os.path.split(i)[-1]
            cv2.imwrite(f'./circle_mask/{fname}_{n}_h.{fextension}', masked)
            n += 1 
        imageList.clear()
        listBox.delete(0,'end')

    # Circle Mask Custom
    def previewCircleMaskCustom(self):
        radius = askinteger("Circle Radius", "Value:", parent=self)
        cX = askinteger("Circle X Position", "Value:", parent=self)
        cY = askinteger("Circle Y Position", "Value:", parent=self)
        n = 0
        for i in imageList:
            image = cv2.imread(i)            
            mask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.circle(mask, (cX, cY), radius, 255, -1)
            masked = cv2.bitwise_and(image, image, mask = mask)
            n += 1 
            cv2.imshow("Applied Mask", masked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Circle Mask Centralized
    def previewCircleMaskCentralized(self):
        radius = askinteger("Circle Radius", "Value:", parent=self)
        n = 0
        for i in imageList:
            image = cv2.imread(i)            
            (cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
            mask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.circle(mask, (cX, cY), radius, 255, -1)
            masked = cv2.bitwise_and(image, image, mask = mask)
            n += 1 
            cv2.imshow("Mask", mask)            
            cv2.imshow("Applied Mask", masked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Drawing Circle Custom preview  
    def previewDrawingCircleCustom(self):
        size = askinteger("Size", "Value:", parent=self)
        radius = askinteger("Circle Radius", "Value:", parent=self)
        cX = askinteger("Circle X Position", "Value:", parent=self)
        cY = askinteger("Circle Y Position", "Value:", parent=self)
        # Creating a black image with 3
        # channels RGB and unsigned int datatype
        image = np.zeros((size, size, 3), dtype = "uint8")
        # Creating circle
        cv2.circle(image, (cX, cY), radius, (255, 255, 255), -1) 
        cv2.imshow('dark', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()       


    # Drawing Circle Centralized preview  
    def previewDrawingCircleCentralized(self):
        size = askinteger("Size", "Value:", parent=self)
        radius = askinteger("Circle Radius", "Value:", parent=self)
        # Creating a black image with 3
        # channels RGB and unsigned int datatype
        image = np.zeros((size, size, 3), dtype = "uint8")
        (cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
        # Creating circle
        cv2.circle(image, (cX, cY), radius, (255, 255, 255), -1)       
        cv2.imshow('dark', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Drawing Circle Custom
    def drawingCircleCustom(self):
        r = random()
        size = askinteger("Size", "Value:", parent=self)
        radius = askinteger("Circle Radius", "Value:", parent=self)
        cX = askinteger("Circle X Position", "Value:", parent=self)
        cY = askinteger("Circle Y Position", "Value:", parent=self)
        # Creating a black image with 3
        # channels RGB and unsigned int datatype
        image = np.zeros((size, size, 3), dtype = "uint8")
        # Creating circle
        circle = cv2.circle(image, (cX, cY), radius, (255, 255, 255), -1)
        cv2.imwrite(f'./geometry_circles/circle_jpg_' + str(size) + '-' + str(r) + '.jpg', circle)
        cv2.imwrite(f'./geometry_circles/circle_png_' + str(size) + '-' + str(r) + '.png', circle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Drawing Circle Centralized
    def drawingCircleCentralized(self):
        r = random()
        size = askinteger("Size", "Value:", parent=self)
        radius = askinteger("Circle Radius", "Value:", parent=self)
        # Creating a black image with 3
        # channels RGB and unsigned int datatype
        image = np.zeros((size, size, 3), dtype = "uint8")
        (cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
        # Creating circle
        circle = cv2.circle(image, (cX, cY), radius, (255, 255, 255), -1)
        cv2.imwrite(f'./geometry_circles/circle_jpg_' + str(size) + '-' + str(r) + '.jpg', circle)
        cv2.imwrite(f'./geometry_circles/circle_png_' + str(size) + '-' + str(r) + '.png', circle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Drawing Rectangle Custom
    def previewRectangleCustom(self):
        size = askinteger("Size", "Value:", parent=self)
        sideX = askinteger("Length", "Value:", parent=self)
        sideY = askinteger("Height", "Value:", parent=self)
        cX = askinteger("X Coordinates", "Value:", parent=self)
        cY = askinteger("Y Coordinates", "Value:", parent=self)
        # Creating a black image with 3
        # channels RGB and unsigned int datatype
        image = np.zeros((size, size, 3), dtype = "uint8")     
        # Creating rectangle
        cv2.rectangle(image, (cX - sideX, cY - sideY), (cX + sideX , cY + sideY), (255, 255, 255), -1)     
        cv2.imshow('Rectangle Preview', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

    # Drawing Rectangle Centralized preview
    def previewRectangleCentralized(self):
        size = askinteger("Size", "Value:", parent=self)
        sideX = askinteger("Length", "Value:", parent=self)
        sideY = askinteger("Height", "Value:", parent=self)
        # Creating a black image with 3
        # channels RGB and unsigned int datatype
        image = np.zeros((size, size, 3), dtype = "uint8")   
        # Creating rectangle
        (cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
        cv2.rectangle(image, (cX - sideX, cY - sideY), (cX + sideX , cY + sideY), (255,255,255), -1)     
        cv2.imshow('dark', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rectangleCustom(self):
        r = random()
        size = askinteger("Size", "Value:", parent=self)
        sideX = askinteger("Length", "Value:", parent=self)
        sideY = askinteger("Height", "Value:", parent=self)
        cX = askinteger("X Coordinates", "Value:", parent=self)
        cY = askinteger("Y Coordinates", "Value:", parent=self)
        # Creating a black image with 3
        # channels RGB and unsigned int datatype
        image = np.zeros((size, size, 3), dtype = "uint8") 
        # Creating rectangle
        rectangle = cv2.rectangle(image, (cX - sideX, cY - sideY), (cX + sideX , cY + sideY), (255, 255, 255), -1)
        cv2.imwrite(f'./geometry_rectangles/rectangle_jpg_' + str(size) + '-' + str(r) + '.jpg', rectangle)
        cv2.imwrite(f'./geometry_rectangles/rectangle_png_' + str(size) + '-' + str(r) + '.png', rectangle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def rectangleCentralized(self):
        r = random()
        size = askinteger("Size", "Value:", parent=self)
        sideX = askinteger("Length", "Value:", parent=self)
        sideY = askinteger("Height", "Value:", parent=self)
        # Creating a black image with 3
        # channels RGB and unsigned int datatype
        image = np.zeros((size, size, 3), dtype = "uint8")        
        # Creating rectangle
        (cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
        rectangle = cv2.rectangle(image, (cX - sideX, cY - sideY), (cX + sideX , cY + sideY), (255,255,255), -1)
        cv2.imwrite(f'./geometry_rectangles/rectangle_jpg_' + str(size) + '-' + str(r) + '.jpg', rectangle)
        cv2.imwrite(f'./geometry_rectangles/rectangle_png_' + str(size) + '-' + str(r) + '.png', rectangle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Rectangle Mask Custom preview
    def previewRectangleMaskCustom(self):
        sideX = askinteger("Rectangle Side X", "Value:", parent=self)
        sideY = askinteger("Rectangle Side Y", "Value:", parent=self)
        cX = askinteger("Rectangle X Position", "Value:", parent=self)
        cY = askinteger("Rectangle Y Position", "Value:", parent=self)
        n = 0
        for i in imageList:
            image = cv2.imread(i)            
            mask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(mask, (cX - sideX, cY - sideY), (cX + sideX , cY + sideY), 255, -1)
            cv2.imshow("Mask", mask)
            # Apply mask
            masked = cv2.bitwise_and(image, image, mask = mask)
            n += 1 
            cv2.imshow("Applied Mask", masked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Rectangle Mask Centralized preview
    def previewRectangleMaskCentralized(self):
        sideX = askinteger("Rectangle Side X", "Value:", parent=self)
        sideY = askinteger("Rectangle Side Y", "Value:", parent=self)
        n = 0
        for i in imageList:
            image = cv2.imread(i)            
            mask = np.zeros(image.shape[:2], dtype = "uint8")
            (cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
            cv2.rectangle(mask, (cX - sideX, cY - sideY), (cX + sideX , cY + sideY), 255, -1)
            cv2.imshow("Mask", mask)
            # Apply mask
            masked = cv2.bitwise_and(image, image, mask = mask)
            n += 1 
            cv2.imshow("Applied Mask", masked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Rectangle Mask Custom
    def rectangleMaskCustom(self):
        sideX = askinteger("Rectangle Side X", "Value:", parent=self)
        sideY = askinteger("Rectangle Side Y", "Value:", parent=self)
        cX = askinteger("Rectangle X Position", "Value:", parent=self)
        cY = askinteger("Rectangle Y Position", "Value:", parent=self)
        n = 0
        for i in imageList:
            image = cv2.imread(i)            
            mask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(mask, (cX - sideX, cY - sideY), (cX + sideX , cY + sideY), 255, -1)
            # Apply mask
            masked = cv2.bitwise_and(image, image, mask = mask)
            fname = os.path.split(i)[1]
            fextension = os.path.split(i)[-1]
            cv2.imwrite(f'./rectangle_mask/{fname}_{n}_h.{fextension}', masked)
            n += 1 
        imageList.clear()
        listBox.delete(0,'end')

    # Rectangle Mask Centralized
    def rectangleMaskCentralized(self):
        sideX = askinteger("Rectangle Side X", "Value:", parent=self)
        sideY = askinteger("Rectangle Side Y", "Value:", parent=self)
        n = 0
        for i in imageList:
            image = cv2.imread(i)            
            mask = np.zeros(image.shape[:2], dtype = "uint8")
            (cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
            cv2.rectangle(mask, (cX - sideX, cY - sideY), (cX + sideX , cY + sideY), 255, -1)
            # Apply mask
            masked = cv2.bitwise_and(image, image, mask = mask)
            fname = os.path.split(i)[1]
            fextension = os.path.split(i)[-1]
            cv2.imwrite(f'./rectangle_mask/{fname}_{n}_h.{fextension}', masked)
            n += 1 
        imageList.clear()
        listBox.delete(0,'end')


    def preview(self):
        # Bitwise AND preview  
        if function_name.get() == 'Bitwise AND':
            comb = np.array(rSubset(imageList, 2))
            n = 0        
            for i in imageList:
                while n < len(comb):           
                    img1 =  cv2.imread(comb[(n,0)])
                    img2 =  cv2.imread(comb[(n,1)])
                    w = int(img1.shape[1])
                    h = int(img1.shape[0])            
                    #Images have to be of the same size to be added
                    #so resize one image to the size of the other before adding
                    img1=cv2.resize(img1, (w,h))
                    img2=cv2.resize(img2, (w,h))
                    bitwiseAnd = cv2.bitwise_and(img1, img2) 
                    n += 1            
                    cv2.imshow('Bitwise AND Preview', bitwiseAnd)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        
        # Bitwise NOT preview 
        if function_name.get() == 'Bitwise NOT':
            n = 0        
            for i in imageList:  
                image = cv2.imread(i)
                bitwiseNot = cv2.bitwise_not(image) 
                n += 1            
                cv2.imshow('Bitwise NOT Preview', bitwiseNot)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Bitwise OR preview  
        if function_name.get() == 'Bitwise OR':
            comb = np.array(rSubset(imageList, 2))
            n = 0        
            for i in imageList:  
                while n < len(comb):                     
                    img1 =  cv2.imread(comb[(n,0)])
                    img2 =  cv2.imread(comb[(n,1)])
                    w = int(img1.shape[1])
                    h = int(img1.shape[0])            
                    #Images have to be of the same size to be added
                    #so resize one image to the size of the other before adding
                    img1=cv2.resize(img1, (w,h))
                    img2=cv2.resize(img2, (w,h))
                    bitwiseOr = cv2.bitwise_or(img1, img2) 
                    n += 1            
                    cv2.imshow('Bitwise OR Preview', bitwiseOr)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        # Bitwise XOR preview
        if function_name.get() == 'Bitwise XOR':
            comb = np.array(rSubset(imageList, 2))
            n = 0        
            for i in imageList:  
                while n < len(comb):                     
                    img1 =  cv2.imread(comb[(n,0)])
                    img2 =  cv2.imread(comb[(n,1)])
                    w = int(img1.shape[1])
                    h = int(img1.shape[0])            
                    #Images have to be of the same size to be added
                    #so resize one image to the size of the other before adding
                    img1=cv2.resize(img1, (w,h))
                    img2=cv2.resize(img2, (w,h))
                    bitwiseXor = cv2.bitwise_xor(img1, img2) 
                    n += 1            
                    cv2.imshow('Bitwise XOR Preview', bitwiseXor)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        # Blending preview
        if function_name.get() == 'Blending':
            p = 0.75
            comb = np.array(rSubset(imageList, 2))
            weight1 = askfloat("First Image Weight", "Value:", parent=self)
            weight2 = askfloat("Second Image Weight", "Value:", parent=self)
            n = 0        
            for i in imageList:   
                while n < len(comb):         
                    img1 =  cv2.imread(comb[(n,0)])
                    img2 =  cv2.imread(comb[(n,1)])
                    #Images have to be of the same size to be added
                    #so resize one image to the size of the other before adding
                    w = int(img1.shape[1] * p)
                    h = int(img1.shape[0] * p)
                    img1=cv2.resize(img1, (w,h))
                    img2=cv2.resize(img2, (w,h))
                    blended = cv2.addWeighted(img1,weight1,img2,weight2,0)
                    n += 1            
                    cv2.imshow('Blending Preview', blended)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        # Blur preview
        if function_name.get() == 'Blur':
            kernel = askinteger("Kernel Size", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                blur_image = cv2.blur(image, (kernel,kernel))
                n += 1 
                cv2.imshow('Blur Preview', blur_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Box Filter preview  
        if function_name.get() == 'Box Filter':
            kernel = askinteger("Kernel Size", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                box_image = cv2.boxFilter(image, -1, (kernel,kernel))
                n += 1 
                cv2.imshow('Box Filter Preview', box_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Circle Mask preview  
        if function_name.get() == 'Circle Mask':
            root = Tk()
            root.wm_iconbitmap('assets/img/logo.ico')
            root.title('OK Image - Choose Mask')
            root.geometry("500x100")
            var = IntVar()
            R1 = Radiobutton(root, text="Custom", variable=var, value=1,
                            command=self.previewCircleMaskCustom)
            R1.pack( anchor = W )

            R2 = Radiobutton(root, text="Centralized", variable=var, value=2,
                            command=self.previewCircleMaskCentralized)
            R2.pack( anchor = W )
            label = Label(root)
            label.pack()
            root.mainloop() 

        # Color Histogram preview  
        if function_name.get() == 'Color Histogram':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                channels = cv2.split(image) 
                colors = ("b", "g", "r") 

                # Histogram
                plt.figure() 
                plt.title("Color Histogram") 
                plt.xlabel("Intensity") 
                plt.ylabel("Total Pixels") 
                for (channels, colors) in zip(channels, colors):
                    # 3 times loop, once for each channel
                    hist = cv2.calcHist([channels], [0], None, [256], [0, 256]) 
                    plt.plot(hist, color = colors) 
                    plt.xlim([0, 256]) 
                plt.show()
                n += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Contours preview  
        if function_name.get() == 'Contours':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                #reduces the contours when blurring the image
                blur = cv2.GaussianBlur(image,(7,7),cv2.BORDER_DEFAULT)
                canny = cv2.Canny(blur,125,175)

                #returns all the contours in the forma of list(RETR_LIST).
                #chain_approx_simple compresses the contorurs and returns only the two end points.
                contours,hierarchies = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                #print("NO of contours : ",len(contours))

                #threshold = binarizing the image.It is another method to find contours
                ret,thresh = cv2.threshold(image,125,125,cv2.THRESH_BINARY)            
                cv2.imshow("Dilated Image",thresh)
                n += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Contrast Stretching preview  
        if function_name.get() == 'Contrast Stretching':
            black = askfloat("Black Point", "Value:", parent=self)
            white = askfloat("White Point", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                contrast_stretch_img = contrast_stretch(image, black, white)
                n += 1 
                cv2.imshow('Contrast Stretching Preview', contrast_stretch_img) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Crop preview  
        if function_name.get() == 'Crop':
            y = askinteger("Crop image y", "Start Y (y):", parent=self)
            yh = askinteger("Crop image y + h", "End Y (y + h):", parent=self)
            x = askinteger("Crop image x", "Start X (x):", parent=self)
            xw = askinteger("Crop image x + w", "End X (x + w):", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                crop_image = image[y:yh, x:xw]
                n += 1 
                cv2.imshow('Crop Preview', crop_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Dilation preview  
        if function_name.get() == 'Dilation':
            kernelx = askinteger("Kernel size", "Value:", parent=self)
            kernely = askinteger("Kernel size", "Value:", parent=self)
            iterations = askinteger("Iterations", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                # dilating the images:
                kernel = np.ones((kernelx,kernely), 'uint8')
                dilated = cv2.dilate(image,kernel,iterations)
                cv2.imshow("Dilated Image",dilated)
                n += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # DoG preview  
        if function_name.get() == 'Dog':
            radius = askinteger("Radis", "Value:", parent=self)
            sigma1 = askinteger("Sigma 1", "Value:", parent=self)
            sigma2 = askinteger("Sigma 2", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                dog_img = dog(image, radius, sigma1, sigma2)
                n += 1 
                cv2.imshow('Difference of Gaussian Preview', dog_img) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Drawing Circle preview  
        if function_name.get() == 'Drawing Circle':
            root = Tk()
            root.wm_iconbitmap('assets/img/logo.ico')
            root.title('OK Image - Drawing Circle')
            root.geometry("500x100")
            var = IntVar()
            R1 = Radiobutton(root, text="Custom", variable=var, value=1,
                            command=self.previewDrawingCircleCustom)
            R1.pack( anchor = W )

            R2 = Radiobutton(root, text="Centralized", variable=var, value=2,
                            command=self.previewDrawingCircleCentralized)
            R2.pack( anchor = W )
            label = Label(root)
            label.pack()
            
        # Drawing Image preview  
        if function_name.get() == 'Drawing Image':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                for y in range(0, image.shape[0], 10):
                    for x in range(0, image.shape[1], 10):
                        image[y:y+5, x: x+5] = (0,255,255)
                n += 1
                cv2.imshow("Modified Image", image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Drawing Rectangle preview  
        if function_name.get() == 'Drawing Rectangle':
            root = Tk()
            root.wm_iconbitmap('assets/img/logo.ico')
            root.title('OK Image - Drawing Rectangle')
            root.geometry("500x100")
            var = IntVar()
            R1 = Radiobutton(root, text="Custom", variable=var, value=1,
                            command=self.previewRectangleCustom)
            R1.pack( anchor = W )

            R2 = Radiobutton(root, text="Centralized", variable=var, value=2,
                            command=self.previewRectangleCentralized)
            R2.pack( anchor = W )
            label = Label(root)
            label.pack()
            root.mainloop()

        # Drawing Text preview  
        if function_name.get() == 'Drawing Text':
            text = askstring("Text", "Value:", parent=self)
            x = askinteger("X Position", "Value:", parent=self)
            y = askinteger("Y Position", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fonte = cv2.FONT_HERSHEY_SIMPLEX 
                cv2.putText(image,text,(x,y), fonte, 2,(255,255,255),2,cv2.LINE_AA)
                n += 1
                cv2.imshow("Modified Image", image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # DType preview  
        if function_name.get() == 'DType':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                dtype_image = image.dtype
                n += 1
                showinfo(title='Data Type', message=dtype_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Edge Cascade preview  
        if function_name.get() == 'Edge Cascade':
            threshold1 = askinteger("High threshold", "Value:", parent=self)
            threshold2 = askinteger("Low threshold", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                canny = cv2.Canny(image,threshold1,threshold2)
                cv2.imshow("Edge Cascade",canny)
                n += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Enhancement preview  
        if function_name.get() == 'Enhancement':
            contrast = askinteger("Contrast", "Alpha Value:", parent=self)  
            brightness = askinteger("Brightness", "Beta Value:", parent=self)        
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                enhanced_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
                n += 1 
                cv2.imshow('Enhancement Preview', enhanced_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Erosion preview  
        if function_name.get() == 'Erosion':
            kernelx = askinteger("Kernel size", "Value:", parent=self)
            kernely = askinteger("Kernel size", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                # eroding the images:
                # Creating kernel
                kernel = np.ones((kernelx,kernely), 'uint8')
                
                # Using cv2.erode() method 
                eroded_image = cv2.erode(image, kernel, cv2.BORDER_REFLECT) 
                cv2.imshow("Eroded Image", eroded_image)
                n += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Flipping preview  
        if function_name.get() == 'Flipping':
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                # Horizontal Flip
                hflipped = cv2.flip(image, 1)            

                # Vertical Flip
                vflipped = cv2.flip(image, 0)            

                # Horizontal and Vertical Flip
                hvflipped = cv2.flip(image, -1)
                n += 1 
                cv2.imshow("Horizontal Flip", hflipped)
                cv2.imshow("Vertical Flip", vflipped)
                cv2.imshow('Horizontal and Vertical Flip', hvflipped)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Gamma Correction preview  
        if function_name.get() == 'Gamma Correction':
            gamma_value = askfloat("Gamma", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                gamma_img = gamma(image, gamma_value)
                n += 1 
                cv2.imshow('Gamma Correction Preview', gamma_img) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Gaussian Blur preview  
        if function_name.get() == 'Gaussian Blur':
            kernel_size = askinteger("Kernel Size", "Value:", parent=self)
            sigma = askinteger("Sigma", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                blur_img = fast_gaussian_blur(image, kernel_size, sigma)
                n += 1 
                cv2.imshow('Gaussian Blur Preview', blur_img) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Grayscale Histogram preview  
        if function_name.get() == 'Grayscale Histogram':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                # Convert from RGB to Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

                # Calculate the histogram
                hist = cv2.calcHist([image], [0], None, [256], [0, 256]) 

                # Histogram
                plt.figure() 
                plt.title("Grayscale Histogram") 
                plt.xlabel("Intensity") 
                plt.ylabel("Total Pixels") 
                plt.plot(hist) 
                plt.xlim([0, 256]) 
                plt.show() 
                n += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Grayscale preview  
        if function_name.get() == 'Grayscale':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                n += 1 
                cv2.imshow('GrayScale Preview', gray_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Horizontal Concat preview  
        if function_name.get() == 'Horizontal Concat':
            n = 0
            comb = np.array(rSubset(imageList, 2))
            for i in imageList:
                while n < len(comb):
                    img1 =  cv2.imread(comb[(n,0)])
                    img2 =  cv2.imread(comb[(n,1)])
                    hconcat_image = hconcat_resize([img1, img2])
                    n += 1            
                    cv2.imshow('Horizontal Concatenation Preview', hconcat_image) 
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        # Histogram Equalization preview  
        if function_name.get() == 'Histogram Equalization':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                # Convert from RGB to Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

                # Histogram equalization
                h_eq = cv2.equalizeHist(image)

                # Images
                cv2.imshow("Original", image)
                cv2.imshow("Equalizated", h_eq)
                cv2.waitKey(0)

                # Histograms
                plt.figure()
                plt.title("Original Histogram") 
                plt.xlabel("Intensity") 
                plt.ylabel("Total Pixels") 
                plt.hist(image.ravel(), 256, [0,256]) 
                plt.xlim([0, 256]) 
                plt.show() 

                plt.figure() 
                plt.title("Equalized Histogram") 
                plt.xlabel("Intensity") 
                plt.ylabel("Total Pixels") 
                plt.hist(h_eq.ravel(), 256, [0,256]) 
                plt.xlim([0, 256]) 
                plt.show() 
                n += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # HSV preview  
        if function_name.get() == 'HSV':
            hmin = askinteger("HMin", "Value:", parent=self)
            smin = askinteger("SMin", "Value:", parent=self)
            vmin = askinteger("VMin", "Value:", parent=self)
            hmax = askinteger("HMax", "Value:", parent=self)
            smax = askinteger("SMax", "Value:", parent=self)
            vmax = askinteger("VMax", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv,(hmin, smin, vmin), (hmax, smax, vmax) )
                n += 1 
                cv2.imshow('HSV Color Preview', mask) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Luminosity Add preview  
        if function_name.get() == 'Luminosity Add':
            value = askinteger("Add Luminosity", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                M = np.ones(image.shape, dtype = "uint8") * value
                added = cv2.add(image, M)
                n += 1 
                cv2.imshow("Luminosity Added", added)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Luminosity Subtract preview  
        if function_name.get() == 'Luminosity Subtract':
            value = askinteger("Subtract Luminosity", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                M = np.ones(image.shape, dtype = "uint8") * value
                subtracted = cv2.subtract(image, M)
                n += 1 
                cv2.imshow("Luminosity Subtracted", subtracted)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Median Blur preview  
        if function_name.get() == 'Median Blur':
            noise = askinteger("Noise", "Value (e.g.: 1, 3, 5, 7 ...):", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                mblur_image = cv2.medianBlur(image,noise)
                n += 1 
                cv2.imshow('Median Blur Preview', mblur_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Merging RGB preview  
        if function_name.get() == 'Merging RGB':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                (B, G, R) = cv2.split(image)
                zeros = np.zeros(image.shape[:2], dtype = "uint8")            
                n += 1 
                cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
                cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
                cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))

                cv2.imshow("Green & Red", cv2.merge([zeros, G, R]))
                cv2.imshow("Blue & Green", cv2.merge([B, G, zeros]))
                cv2.imshow("Blue & Red", cv2.merge([B, zeros, R]))

                merged = cv2.merge([B, G, R])
                cv2.imshow("Merged", merged)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Negative preview  
        if function_name.get() == 'Negative':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                negative_img = negate(image)
                n += 1 
                cv2.imshow('Negative Image Preview', negative_img) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Normalization preview  
        if function_name.get() == 'Normalization':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                norm = np.zeros((800,800))
                norm_image = cv2.normalize(image,norm,0,255,cv2.NORM_MINMAX)
                cv2.imshow("Low Quality Image",image)
                cv2.imshow("Normalized Image",norm_image)
                n += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Number Of Contours preview  
        if function_name.get() == 'Number Of Contours':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                blur = cv2.GaussianBlur(image,(7,7),cv2.BORDER_DEFAULT)
                canny = cv2.Canny(blur,125,175)

                #returns all the contours in the forma of list(RETR_LIST).
                #chain_approx_simple compresses the contorurs and returns only the two end points.
                contours,hierarchies = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                n_contours = len(contours)
                n += 1
                showinfo(title='Number of Contours', message=n_contours) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Pixel Values preview  
        if function_name.get() == 'Pixel Values':
            width = askinteger("Row", "Row Number:", parent=self)
            height = askinteger("Column", "Column Number:", parent=self)
            n = 0
            for i in imageList:
                image = Image.open(i)
                pixels_image = image.getpixel((width, height))
                n += 1
                showinfo(title='Pixel', message=pixels_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Rectangle Mask preview  
        if function_name.get() == 'Rectangle Mask':
            root = Tk()
            root.wm_iconbitmap('assets/img/logo.ico')
            root.title('OK Image - Choose Mask')
            root.geometry("500x100")
            var = IntVar()
            R1 = Radiobutton(root, text="Custom", variable=var, value=1,
                            command=self.previewRectangleMaskCustom)
            R1.pack( anchor = W )

            R2 = Radiobutton(root, text="Centralized", variable=var, value=2,
                            command=self.previewRectangleMaskCentralized)
            R2.pack( anchor = W )
            label = Label(root)
            label.pack()
            root.mainloop() 

        # Resize preview  
        if function_name.get() == 'Resize':
            width = askinteger("Resize image width", "Width:", parent=self)
            height = askinteger("Resize image height", "Height:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                resize_image = cv2.resize(image, (width,height))
                n += 1 
                cv2.imshow('Resize Preview', resize_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Rotate 45 Degrees preview  
        if function_name.get() == 'Rotate 45 Degrees':
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                # image dimensions and calculate the image's center
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                # rotate image 45 degrees
                M = cv2.getRotationMatrix2D(center, 45, 1.0)
                rotated_45_image = cv2.warpAffine(image, M, (w, h))
                n += 1 
                cv2.imshow('Rotate 45º Preview', rotated_45_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
        # Rotate 90 Degrees preview  
        if function_name.get() == 'Rotate 90 Degrees':
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                rotated_90_image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
                n += 1 
                cv2.imshow('Rotate 90º Preview', rotated_90_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Rotate 180 Degrees preview  
        if function_name.get() == 'Rotate 180 Degrees':
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                rotated_180_image = cv2.rotate(image, cv2.cv2.ROTATE_180)
                n += 1 
                cv2.imshow('Rotate 180º Preview', rotated_180_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Rotate 270 Degrees preview  
        if function_name.get() == 'Rotate 270 Degrees':
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                rotated_270_image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                n += 1 
                cv2.imshow('Rotate 270º Preview', rotated_270_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Shape preview  
        if function_name.get() == 'Shape':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                dtype_image = image.shape
                n += 1
                showinfo(title='Shape', message=dtype_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Shift preview  
        if function_name.get() == 'Shift':
            x = askinteger("X axis shift", "Value:", parent=self)
            y = askinteger("Y axis shift", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                M = np.float32([[1, 0, x], [0, 1, y]])
                shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                n += 1 
                cv2.imshow('Shift Preview', shifted) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Splitting RGB preview  
        if function_name.get() == 'Splitting RGB':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                (B, G, R) = cv2.split(image)
                zeros = np.zeros(image.shape[:2], dtype = "uint8")
                n += 1 
                cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
                cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
                cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Vertical Concat preview  
        if function_name.get() == 'Vertical Concat':
            n = 0
            comb = np.array(rSubset(imageList, 2))
            for i in imageList:
                while n < len(comb):         
                    img1 =  cv2.imread(comb[(n,0)])
                    img2 =  cv2.imread(comb[(n,1)])
                    vconcat_image = vconcat_resize([img1, img2])            
                    n += 1          
                    cv2.imshow('Vertical Concatenation Preview', vconcat_image) 
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        # Vertical And Horizontal Concat preview  
        if function_name.get() == 'Vertical And Horizontal Concat':
            n = 0
            for i in imageList:            
                image = cv2.imread(i)
                # image resizing
                img1_s = cv2.resize(image, dsize = (0,0),
                                    fx = 0.5, fy = 0.5)                
                # function calling
                vhconcat_image = concat_vh([[img1_s, img1_s, img1_s],
                                    [img1_s, img1_s, img1_s],
                                    [img1_s, img1_s, img1_s]])
                n += 1            
                cv2.imshow('Vertical and Horizontal Concatenation Preview', vhconcat_image) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Zooming preview  
        if function_name.get() == 'Zooming':
            y = askinteger("Crop image y", "Start Y (y):", parent=self)
            yh = askinteger("Crop image y + h", "End Y (y + h):", parent=self)
            x = askinteger("Crop image x", "Start X (x):", parent=self)
            xw = askinteger("Crop image x + w", "End X (x + w):", parent=self)
            zoom_factor = askinteger("Zoom Factor", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                crop_image = image[y:yh, x:xw]
                zoom_image = cv2.resize(crop_image, None, fx=zoom_factor, fy=zoom_factor)
                cv2.imshow("Zooming",zoom_image)
                n += 1
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            
    def save(self):
        # Bitwise AND save in batch
        if function_name.get() == 'Bitwise AND':
            comb = np.array(rSubset(imageList, 2))
            n = 0
            for i in imageList:
                while n < len(comb):
                    image = cv2.imread(i)
                    fname = os.path.split(i)[1]
                    fextension = os.path.split(i)[-1]
                    img1 =  cv2.imread(comb[(n,0)])
                    img2 =  cv2.imread(comb[(n,1)])
                    w = int(img1.shape[1])
                    h = int(img1.shape[0])            
                    #Images have to be of the same size to be added
                    #so resize one image to the size of the other before adding
                    img1=cv2.resize(img1, (w,h))
                    img2=cv2.resize(img2, (w,h))
                    bitwiseAnd = cv2.bitwise_and(img1, img2)
                    cv2.imwrite(f'./bitwise_AND/{fname}_{n}.{fextension}', bitwiseAnd)
                    n += 1              
                imageList.clear()
                listBox.delete(0,'end') # clear listbox

        # Bitwise NOT save in batch
        if function_name.get() == 'Bitwise NOT':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                bitwiseNot = cv2.bitwise_not(image)
                cv2.imwrite(f'./bitwise_NOT/{fname}_{n}.{fextension}', bitwiseNot)
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Bitwise OR save in batch  
        if function_name.get() == 'Bitwise OR':
            comb = np.array(rSubset(imageList, 2))
            n = 0
            for i in imageList:
                while n < len(comb):           
                    image = cv2.imread(i)
                    fname = os.path.split(i)[1]
                    fextension = os.path.split(i)[-1]
                    img1 =  cv2.imread(comb[(n,0)])
                    img2 =  cv2.imread(comb[(n,1)])
                    w = int(img1.shape[1])
                    h = int(img1.shape[0])            
                    #Images have to be of the same size to be added
                    #so resize one image to the size of the other before adding
                    img1=cv2.resize(img1, (w,h))
                    img2=cv2.resize(img2, (w,h))
                    bitwiseOr = cv2.bitwise_or(img1, img2)
                    cv2.imwrite(f'./bitwise_OR/{fname}_{n}.{fextension}', bitwiseOr)
                    n += 1              
                imageList.clear()
                listBox.delete(0,'end')

        # Bitwise XOR save in batch  
        if function_name.get() == 'Bitwise XOR':
            comb = np.array(rSubset(imageList, 2))
            n = 0
            for i in imageList:
                while n < len(comb):           
                    image = cv2.imread(i)
                    fname = os.path.split(i)[1]
                    fextension = os.path.split(i)[-1]
                    img1 =  cv2.imread(comb[(n,0)])
                    img2 =  cv2.imread(comb[(n,1)])
                    w = int(img1.shape[1])
                    h = int(img1.shape[0])            
                    #Images have to be of the same size to be added
                    #so resize one image to the size of the other before adding
                    img1=cv2.resize(img1, (w,h))
                    img2=cv2.resize(img2, (w,h))
                    bitwiseXor = cv2.bitwise_xor(img1, img2)
                    cv2.imwrite(f'./bitwise_XOR/{fname}_{n}.{fextension}', bitwiseXor)
                    n += 1              
                imageList.clear()
                listBox.delete(0,'end')

        # Blending save in batch  
        if function_name.get() == 'Blending':
            p = 0.75
            comb = np.array(rSubset(imageList, 2))
            weight1 = askfloat("First Image Weight", "Value:", parent=self)
            weight2 = askfloat("Second Image Weight", "Value:", parent=self)
            n = 0
            for i in imageList:
                while n < len(comb):
                    image = cv2.imread(i)
                    fname = os.path.split(i)[1]
                    fextension = os.path.split(i)[-1]
                    img1 =  cv2.imread(comb[(n,0)])
                    img2 =  cv2.imread(comb[(n,1)])
                    w = int(img1.shape[1] * p)
                    h = int(img1.shape[0] * p)            
                    #Images have to be of the same size to be added
                    #so resize one image to the size of the other before adding
                    img1=cv2.resize(img1, (w,h))
                    img2=cv2.resize(img2, (w,h))
                    blended = cv2.addWeighted(img1,weight1,img2,weight2,0)
                    blended_2 = cv2.addWeighted(img2,weight1,img1,weight2,0)
                    cv2.imwrite(f'./blended/{fname}_{n}.{fextension}', blended)
                    cv2.imwrite(f'./blended/{fname}_{n}_inverse.{fextension}', blended_2)
                    n += 1              
                imageList.clear()
                listBox.delete(0,'end')

        # Blur save in batch  
        if function_name.get() == 'Blur':
            kernel = askinteger("Kernel Size", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                blur_image = cv2.blur(image, (kernel,kernel))
                cv2.imwrite(f'./blured/{fname}_{n}.{fextension}', blur_image)
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Box Filter save in batch  
        if function_name.get() == 'Box Filter':
            kernel = askinteger("Kernel Size", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                box_image = cv2.boxFilter(image, -1, (kernel,kernel))
                cv2.imwrite(f'./box_filtered/{fname}_{n}.{fextension}', box_image)
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Circle Mask save in batch  
        if function_name.get() == 'Circle Mask':
            root = Tk()
            root.wm_iconbitmap('assets/img/logo.ico')
            root.title('OK Image - Choose Mask')
            root.geometry("500x100")
            var = IntVar()
            R1 = Radiobutton(root, text="Custom", variable=var, value=1,
                            command=self.circleMaskCustom)
            R1.pack( anchor = W )

            R2 = Radiobutton(root, text="Centralized", variable=var, value=2,
                            command=self.circleMaskCentralized)
            R2.pack( anchor = W )
            label = Label(root)
            label.pack()
            root.mainloop()   

        # Color Histogram save in batch  
        if function_name.get() == 'Color Histogram':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]            
                channels = cv2.split(image) 
                colors = ("b", "g", "r") 

                n += 1 
                plt.figure() 
                plt.title("Color Histogram") 
                plt.xlabel("Intensity") 
                plt.ylabel("Total Pixels") 
                for (channels, colors) in zip(channels, colors):
                    # 3 times loop, once for each channel
                    hist = cv2.calcHist([channels], [0], None, [256], [0, 256]) 
                    plt.plot(hist, color = colors) 
                    plt.xlim([0, 256]) 
                plt.savefig(f'./get_color_histogram/{fname}_{n}.{fextension}')             
            imageList.clear()
            listBox.delete(0,'end')

        # Contours save in batch  
        if function_name.get() == 'Contours':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                #reduces the contours when blurring the image
                blur = cv2.GaussianBlur(image,(7,7),cv2.BORDER_DEFAULT)
                canny = cv2.Canny(blur,125,175)

                #returns all the contours in the forma of list(RETR_LIST).
                #chain_approx_simple compresses the contorurs and returns only the two end points.
                contours,hierarchies = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                #print("NO of contours : ",len(contours))

                #threshold = binarizing the image.It is another method to find contours
                ret,thresh = cv2.threshold(image,125,125,cv2.THRESH_BINARY)            

                cv2.imwrite(f'./contourned/{fname}_{n}.{fextension}', thresh)             
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Contrast Stretching save in batch 
        if function_name.get() == 'Contrast Stretching':
            black = askfloat("Black Point", "Value:", parent=self)
            white = askfloat("White Point", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                contrast_stretch_img = contrast_stretch(image, black, white)
                cv2.imwrite(f'./contrast_stretched/{fname}_{n}.{fextension}', contrast_stretch_img)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Crop save in batch 
        if function_name.get() == 'Crop':
            y = askinteger("Crop image y", "Start Y (y):", parent=self)
            yh = askinteger("Crop image y + h", "End Y (y + h):", parent=self)
            x = askinteger("Crop image x", "Start X (x):", parent=self)
            xw = askinteger("Crop image x + w", "End X (x + w):", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                crop_image = image[y:yh, x:xw]
                cv2.imwrite(f'./cropped/{fname}_{n}.{fextension}', crop_image)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Dilation save in batch  
        if function_name.get() == 'Dilation':
            kernelx = askinteger("Kernel size", "Value:", parent=self)
            kernely = askinteger("Kernel size", "Value:", parent=self)
            iterations = askinteger("Iterations", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                # dilating the images:
                kernel = np.ones((kernelx,kernely), 'uint8')
                dilated = cv2.dilate(image,kernel,iterations)
                cv2.imwrite(f'./dilated/{fname}_{n}.{fextension}', dilated)             
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # DoG save in batch 
        if function_name.get() == 'Dog':
            radius = askinteger("Radis", "Value:", parent=self)
            sigma1 = askinteger("Sigma 1", "Value:", parent=self)
            sigma2 = askinteger("Sigma 2", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                dog_img = dog(image, radius, sigma1, sigma2)
                cv2.imwrite(f'./dog/{fname}_{n}.{fextension}', dog_img)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')   

        # Drawing Circle save in batch  
        if function_name.get() == 'Drawing Circle':
            root = Tk()
            root.wm_iconbitmap('assets/img/logo.ico')
            root.title('OK Image - Drawing Circle')
            root.geometry("500x100")
            var = IntVar()
            R1 = Radiobutton(root, text="Custom", variable=var, value=1,
                            command=self.drawingCircleCustom)
            R1.pack( anchor = W )

            R2 = Radiobutton(root, text="Centralized", variable=var, value=2,
                            command=self.drawingCircleCentralized)
            R2.pack( anchor = W )
            label = Label(root)
            label.pack()
            root.mainloop()

        # Drawing Image save in batch  
        if function_name.get() == 'Drawing Image':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                for y in range(0, image.shape[0], 10):
                    for x in range(0, image.shape[1], 10):
                        image[y:y+5, x: x+5] = (0,255,255)
                cv2.imwrite(f'./image_drawed/{fname}_{n}.{fextension}', image)             
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Drawing Rectangle save in batch  
        if function_name.get() == 'Drawing Rectangle':
            root = Tk()
            root.wm_iconbitmap('assets/img/logo.ico')
            root.title('OK Image - Drawing Rectangle')
            root.geometry("500x100")
            var = IntVar()
            R1 = Radiobutton(root, text="Custom", variable=var, value=1,
                            command=self.rectangleCustom)
            R1.pack( anchor = W )

            R2 = Radiobutton(root, text="Centralized", variable=var, value=2,
                            command=self.rectangleCentralized)
            R2.pack( anchor = W )
            label = Label(root)
            label.pack()
            root.mainloop()

        # Drawing Text save in batch  
        if function_name.get() == 'Drawing Text':
            text = askstring("Text", "Value:", parent=self)
            x = askinteger("X Position", "Value:", parent=self)
            y = askinteger("Y Position", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                fonte = cv2.FONT_HERSHEY_SIMPLEX 
                cv2.putText(image,text,(x,y), fonte, 2,(255,255,255),2,cv2.LINE_AA)
                cv2.imwrite(f'./text_drawed/{fname}_{n}.{fextension}', image)             
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # DType save in batch  
        if function_name.get() == 'DType':
            n = 0
            r = random()
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]            
                dtype_image = image.dtype
                n += 1 
                with open(f"./get_dtype/image_dtype_"+str(r)+".txt", "a") as file_object:
                    file_object.write(fname + fextension + " - " + str(dtype_image)) 
                    file_object.write("\n")                      
            imageList.clear()
            listBox.delete(0,'end')
            file_object.close()

        # Edge Cascade save in batch  
        if function_name.get() == 'Edge Cascade':
            threshold1 = askinteger("High threshold", "Value:", parent=self)
            threshold2 = askinteger("Low threshold", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                canny = cv2.Canny(image,threshold1,threshold2)
                cv2.imwrite(f'./edge_cascade/{fname}_{n}.{fextension}', canny)             
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Enhancement save in batch  
        if function_name.get() == 'Enhancement':
            contrast = askinteger("Contrast", "Alpha Value:", parent=self)  
            brightness = askinteger("Brightness", "Beta Value:", parent=self)         
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                enhanced_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
                cv2.imwrite(f'./enhanced/{fname}_{n}.{fextension}', enhanced_image)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Erosion save in batch  
        if function_name.get() == 'Erosion':
            kernelx = askinteger("Kernel size", "Value:", parent=self)
            kernely = askinteger("Kernel size", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                # eroding the images:
                # Creating kernel
                kernel = np.ones((kernelx, kernely), np.uint8)
                
                # Using cv2.erode() method 
                eroded_image = cv2.erode(image, kernel) 
                cv2.imwrite(f'./eroded/{fname}_{n}.{fextension}', eroded_image)             
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Flipping save in batch
        if function_name.get() == 'Flipping':
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                # Horizontal Flip
                hflipped = cv2.flip(image, 1)            

                # Vertical Flip
                vflipped = cv2.flip(image, 0)            

                # Horizontal and Vertical Flip
                hvflipped = cv2.flip(image, -1)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                cv2.imwrite(f'./flipped/{fname}_{n}_h.{fextension}', hflipped)
                cv2.imwrite(f'./flipped/{fname}_{n}_v.{fextension}', vflipped)
                cv2.imwrite(f'./flipped/{fname}_{n}_hv.{fextension}', hvflipped)
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Gamma Correction save in batch
        if function_name.get() == 'Gamma Correction':
            gamma_value = askfloat("Gamma", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                gamma_img = gamma(image, gamma_value)
                cv2.imwrite(f'./gamma_correction/{fname}_{n}.{fextension}', gamma_img)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Gaussian Blur save in batch
        if function_name.get() == 'Gaussian Blur':
            kernel_size = askinteger("Kernel Size", "Value:", parent=self)
            sigma = askinteger("Sigma", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                blur_img = fast_gaussian_blur(image, kernel_size, sigma)
                cv2.imwrite(f'./gaussian_blur/{fname}_{n}.{fextension}', blur_img)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Grayscale Histogram save in batch  
        if function_name.get() == 'Grayscale Histogram':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]            
                # Convert from RGB to Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

                # Calculate the histogram
                hist = cv2.calcHist([image], [0], None, [256], [0, 256]) 

                n += 1 
                plt.figure() 
                plt.title("Grayscale Histogram") 
                plt.xlabel("Intensity") 
                plt.ylabel("Total Pixels") 
                plt.plot(hist) 
                plt.xlim([0, 256]) 
                plt.savefig(f'./get_grayscale_histogram/{fname}_{n}.{fextension}')             
            imageList.clear()
            listBox.delete(0,'end')

        # Grayscale save in batch
        if function_name.get() == 'Grayscale':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(f'./grayscaled/{fname}_{n}.{fextension}', gray_image)             
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Horizontal Concat save in batch 
        if function_name.get() == 'Horizontal Concat':
            n = 0
            comb = np.array(rSubset(imageList, 2))
            for i in imageList:
                while n < len(comb):
                    #image = cv2.imread(i)
                    fname = os.path.split(i)[1]
                    fextension = os.path.split(i)[-1]
                    img1 =  cv2.imread(comb[(n,0)])
                    img2 =  cv2.imread(comb[(n,1)])
                    hconcat_image = hconcat_resize([img1, img2])
                    hconcat_image_2 = hconcat_resize([img2, img1])
                    cv2.imwrite(f'./h_concatenated/{fname}_{n}.{fextension}', hconcat_image)
                    cv2.imwrite(f'./h_concatenated/{fname}_{n}_inverse.{fextension}', hconcat_image_2)
                    n += 1              
                imageList.clear()
                listBox.delete(0,'end')

        # Histogram Equalization save in batch  
        if function_name.get() == 'Histogram Equalization':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]            
                # Convert from RGB to Grayscale
                eq_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

                # Histogram equalization
                h_eq = cv2.equalizeHist(eq_image)

                # Histograms
                plt.figure()
                plt.title("Original Histogram") 
                plt.xlabel("Intensity") 
                plt.ylabel("Total Pixels") 
                plt.hist(image.ravel(), 256, [0,256]) 
                plt.xlim([0, 256]) 
                plt.savefig(f'./get_histogram_equalization/{fname}_{n}_original.{fextension}')             

                plt.figure() 
                plt.title("Equalized Histogram") 
                plt.xlabel("Intensity") 
                plt.ylabel("Total Pixels") 
                plt.hist(h_eq.ravel(), 256, [0,256]) 
                plt.xlim([0, 256]) 
                plt.savefig(f'./get_histogram_equalization/{fname}_{n}_equalized.{fextension}')

                cv2.imwrite(f'./get_histogram_equalization/{fname}_{n}.{fextension}', eq_image)

                n += 1                 
            imageList.clear()
            listBox.delete(0,'end')

        # HSV save in batch 
        if function_name.get() == 'HSV':
            hmin = askinteger("HMin", "Value:", parent=self)
            smin = askinteger("SMin", "Value:", parent=self)
            vmin = askinteger("VMin", "Value:", parent=self)
            hmax = askinteger("HMax", "Value:", parent=self)
            smax = askinteger("SMax", "Value:", parent=self)
            vmax = askinteger("VMax", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv,(hmin, smin, vmin), (hmax, smax, vmax) )
                cv2.imwrite(f'./hsv/{fname}_{n}.{fextension}', mask)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Luminosity Add save in batch 
        if function_name.get() == 'Luminosity Add':
            value = askinteger("Add Luminosity", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                M = np.ones(image.shape, dtype = "uint8") * value
                added = cv2.add(image, M)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                cv2.imwrite(f'./luminosity_added/{fname}_{n}_h.{fextension}', added)
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Luminosity Subtract save in batch
        if function_name.get() == 'Luminosity Subtract':
            value = askinteger("Subtract Luminosity", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                M = np.ones(image.shape, dtype = "uint8") * value
                subtracted = cv2.subtract(image, M)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                cv2.imwrite(f'./luminosity_subtracted/{fname}_{n}_h.{fextension}', subtracted)
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Median Blur save in batch  
        if function_name.get() == 'Median Blur':
            noise = askinteger("Noise", "Value (e.g.: 1, 3, 5, 7 ...):", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                mblur_image = cv2.medianBlur(image,noise)
                cv2.imwrite(f'./median_blured/{fname}_{n}.{fextension}', mblur_image)
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Merging RGB save in batch  
        if function_name.get() == 'Merging RGB':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                (B, G, R) = cv2.split(image)
                zeros = np.zeros(image.shape[:2], dtype = "uint8")
                cv2.imwrite(f'./RGB_merged/{fname}_{n}_R.{fextension}', cv2.merge([zeros, zeros, R]))
                cv2.imwrite(f'./RGB_merged/{fname}_{n}_G.{fextension}', cv2.merge([zeros, G, zeros]))
                cv2.imwrite(f'./RGB_merged/{fname}_{n}_B.{fextension}', cv2.merge([B, zeros, zeros]))

                cv2.imwrite(f'./RGB_merged/{fname}_{n}_GR.{fextension}', cv2.merge([zeros, G, R]))
                cv2.imwrite(f'./RGB_merged/{fname}_{n}_BG.{fextension}', cv2.merge([B, G, zeros]))
                cv2.imwrite(f'./RGB_merged/{fname}_{n}_BR.{fextension}', cv2.merge([B, zeros, R]))

                merged = cv2.merge([B, G, R])
                cv2.imwrite(f'./RGB_merged/{fname}_{n}_Merged.{fextension}', merged)
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Negative save in batch  
        if function_name.get() == 'Negative':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                negative_img = negate(image)
                cv2.imwrite(f'./negative/{fname}_{n}.{fextension}', negative_img)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Normalization save in batch  
        if function_name.get() == 'Normalization':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                norm = np.zeros((800,800))
                norm_image = cv2.normalize(image,norm,0,255,cv2.NORM_MINMAX)
                cv2.imwrite(f'./normalized/{fname}_{n}.{fextension}', norm_image)             
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Number Of Contours save in batch  
        if function_name.get() == 'Number Of Contours':
            n = 0
            r = random()
            for i in imageList:
                image = cv2.imread(i)            
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]            
                blur = cv2.GaussianBlur(image,(7,7),cv2.BORDER_DEFAULT)
                canny = cv2.Canny(blur,125,175)

                #returns all the contours in the forma of list(RETR_LIST).
                #chain_approx_simple compresses the contorurs and returns only the two end points.
                contours,hierarchies = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                n_contours = len(contours)
                n += 1 
                with open(f"./get_number_contours/image_contours_"+str(r)+".txt", "a") as file_object:                
                    file_object.write(fname + fextension + " - Number of Contours : " + str(n_contours)) 
                    file_object.write("\n")                      
            imageList.clear()
            listBox.delete(0,'end') # clear listbox
            file_object.close() 

        # Pixel Values save in batch  
        if function_name.get() == 'Pixel Values':
            width = askinteger("Row", "Row Number:", parent=self)
            height = askinteger("Column", "Column Number:", parent=self)
            n = 0
            r = random()
            for i in imageList:
                image = Image.open(i)            
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]            
                pixels_image = image.getpixel((width, height))
                n += 1 
                with open(f"./get_pixels/image_pixels_"+str(r)+".txt", "a") as file_object:                
                    file_object.write(fname + fextension + " - " + str(pixels_image)) 
                    file_object.write("\n")                      
            imageList.clear()
            listBox.delete(0,'end')
            file_object.close()

        # Rectangle Mask save in batch  
        if function_name.get() == 'Rectangle Mask':
            root = Tk()
            root.wm_iconbitmap('assets/img/logo.ico')
            root.title('OK Image - Choose Mask')
            root.geometry("500x100")
            var = IntVar()
            R1 = Radiobutton(root, text="Custom", variable=var, value=1,
                            command=self.rectangleMaskCustom)
            R1.pack( anchor = W )

            R2 = Radiobutton(root, text="Centralized", variable=var, value=2,
                            command=self.rectangleMaskCentralized)
            R2.pack( anchor = W )
            label = Label(root)
            label.pack()
            root.mainloop()

        # Resize save in batch 
        if function_name.get() == 'Resize':
            width = askinteger("Resize image width", "Width:", parent=self)
            height = askinteger("Resize image height", "Height:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                resize_image = cv2.resize(image, (width, height))
                cv2.imwrite(f'./resized/{fname}_{n}.{fextension}', resize_image)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Rotate 45 Degrees save in batch 
        if function_name.get() == 'Rotate 45 Degrees':
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                # image dimensions and calculate the image's center
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                # rotate image 45 degrees
                M = cv2.getRotationMatrix2D(center, 45, 1.0)
                rotated_45_image = cv2.warpAffine(image, M, (w, h))
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                cv2.imwrite(f'./rotated45/{fname}_{n}.{fextension}', rotated_45_image)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')
            
        # Rotate 90 Degrees save in batch 
        if function_name.get() == 'Rotate 90 Degrees':
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                rotated_90_image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                cv2.imwrite(f'./rotated90/{fname}_{n}.{fextension}', rotated_90_image)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Rotate 180 Degrees save in batch  
        if function_name.get() == 'Rotate 180 Degrees':
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                rotated_180_image = cv2.rotate(image, cv2.cv2.ROTATE_180)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                cv2.imwrite(f'./rotated180/{fname}_{n}.{fextension}', rotated_180_image)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Rotate 270 Degrees save in batch 
        if function_name.get() == 'Rotate 270 Degrees':
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                rotated_270_image = cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                cv2.imwrite(f'./rotated270/{fname}_{n}.{fextension}', rotated_270_image)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Shape save in batch
        if function_name.get() == 'Shape':
            n = 0
            r = random()
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]            
                shape_image = image.shape
                n += 1 
                with open(f"./get_shape/image_shape_"+str(r)+".txt", "a") as file_object:
                    file_object.write(fname + fextension + " - " + str(shape_image)) 
                    file_object.write("\n")                      
            imageList.clear()
            listBox.delete(0,'end')
            file_object.close()

        # Shift save in batch 
        if function_name.get() == 'Shift':
            x = askinteger("X axis shift", "Value:", parent=self)
            y = askinteger("Y axis shift", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)            
                M = np.float32([[1, 0, x], [0, 1, y]])
                shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                cv2.imwrite(f'./shifted/{fname}_{n}.{fextension}', shifted)             
                n += 1 
            imageList.clear()
            listBox.delete(0,'end')

        # Splitting RGB save in batch  
        if function_name.get() == 'Splitting RGB':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                (B, G, R) = cv2.split(image)
                zeros = np.zeros(image.shape[:2], dtype = "uint8")
                cv2.imwrite(f'./RGB_splitted/{fname}_{n}_R.{fextension}', cv2.merge([zeros, zeros, R]))
                cv2.imwrite(f'./RGB_splitted/{fname}_{n}_G.{fextension}', cv2.merge([zeros, G, zeros]))
                cv2.imwrite(f'./RGB_splitted/{fname}_{n}_B.{fextension}', cv2.merge([B, zeros, zeros]))             
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Vertical Concat save in batch  
        if function_name.get() == 'Vertical Concat':
            n = 0
            comb = np.array(rSubset(imageList, 2))
            for i in imageList:
                while n < len(comb):
                    fname = os.path.split(i)[1]
                    fextension = os.path.split(i)[-1]
                    img1 =  cv2.imread(comb[(n,0)])
                    img2 =  cv2.imread(comb[(n,1)])
                    vconcat_image = vconcat_resize([img1, img2])
                    vconcat_image_2 = vconcat_resize([img2, img1])
                    cv2.imwrite(f'./v_concatenated/{fname}_{n}.{fextension}', vconcat_image)
                    cv2.imwrite(f'./v_concatenated/{fname}_{n}_inverse.{fextension}', vconcat_image_2)
                    n += 1              
                imageList.clear()
                listBox.delete(0,'end')

        # Vertical And Horizontal Concat save in batch  
        if function_name.get() == 'Vertical And Horizontal Concat':
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                # image resizing
                img1_s = cv2.resize(image, dsize = (0,0),
                                    fx = 0.5, fy = 0.5)                
                # function calling
                vhconcat_image = concat_vh([[img1_s, img1_s, img1_s],
                                    [img1_s, img1_s, img1_s],
                                    [img1_s, img1_s, img1_s]])
                cv2.imwrite(f'./vh_concatenated/{fname}_{n}.{fextension}', vhconcat_image)             
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')

        # Zooming save in batch 
        if function_name.get() == 'Zooming':
            y = askinteger("Crop image y", "Start Y (y):", parent=self)
            yh = askinteger("Crop image y + h", "End Y (y + h):", parent=self)
            x = askinteger("Crop image x", "Start X (x):", parent=self)
            xw = askinteger("Crop image x + w", "End X (x + w):", parent=self)
            zoom_factor = askinteger("Zoom Factor", "Value:", parent=self)
            n = 0
            for i in imageList:
                image = cv2.imread(i)
                fname = os.path.split(i)[1]
                fextension = os.path.split(i)[-1]
                crop_image = image[y:yh, x:xw]
                zoom_image = cv2.resize(crop_image, None, fx=zoom_factor, fy=zoom_factor)
                cv2.imwrite(f'./zooming/{fname}_{n}.{fextension}', zoom_image)             
                n += 1              
            imageList.clear()
            listBox.delete(0,'end')


    def results(self):
        # bitwise AND results
        if function_name.get() == 'Bitwise AND':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./bitwise_AND", title='Get your images')
            return filename   

        # bitwise NOT results
        if function_name.get() == 'Bitwise NOT':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./bitwise_NOT", title='Get your images')
            return filename

        # Bitwise OR results  
        if function_name.get() == 'Bitwise OR':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./bitwise_OR", title='Get your images')
            return filename

        # Bitwise XOR results  
        if function_name.get() == 'Bitwise XOR':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./bitwise_XOR", title='Get your images')
            return filename

        # Blending results  
        if function_name.get() == 'Blending':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./blended", title='Get your images')
            return filename

        # Blur results  
        if function_name.get() == 'Blur':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./blured", title='Get your images')
            return filename

        # Box Filter results  
        if function_name.get() == 'Box Filter':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./box_filtered", title='Get your images')
            return filename

        # Circle Mask results  
        if function_name.get() == 'Circle Mask':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./circle_mask", title='Get your images')
            return filename

        # Color Histogram results  
        if function_name.get() == 'Color Histogram':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif'))]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./get_color_histogram", title='Get your images data')
            return filename

        # Contours results 
        if function_name.get() == 'Contours':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./contourned", title='Get your images')
            return filename

        # Contrast Stretching results 
        if function_name.get() == 'Contrast Stretching':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./contrast_stretched", title='Get your images')
            return filename

        # Crop results 
        if function_name.get() == 'Crop':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./cropped", title='Get your images')
            return filename

        # Dilation results  
        if function_name.get() == 'Dilation':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./dilated", title='Get your images')
            return filename

        # Dog results 
        if function_name.get() == 'Dog':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./dog", title='Get your images')
            return filename

        # Drawing Circle results  
        if function_name.get() == 'Drawing Circle':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./geometry_circles", title='Get your images')
            return filename

        # Drawing Image results  
        if function_name.get() == 'Drawing Image':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./image_drawed", title='Get your images')
            return filename

        # Drawing Rectangle results
        if function_name.get() == 'Drawing Rectangle':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./geometry_rectangles", title='Get your images')
            return filename

        # Drawing Text results
        if function_name.get() == 'Drawing Text':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./text_drawed", title='Get your images')
            return filename
       
        # DType results
        if function_name.get() == 'DType':
            image_formats= [('TXT', ('*.txt'))]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./get_dtype", title='Get your images data')
            return filename

        # Edge Cascade results 
        if function_name.get() == 'Edge Cascade':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./edge_cascade", title='Get your images')
            return filename

        # Enhancement results
        if function_name.get() == 'Enhancement':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./enhanced", title='Get your images')
            return filename

        # Erosion results  
        if function_name.get() == 'Erosion':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./eroded", title='Get your images')
            return filename

        # Flipping results
        if function_name.get() == 'Flipping':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./flipped", title='Get your images')
            return filename

        # Gamma Correction results
        if function_name.get() == 'Gamma Correction':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./gamma_correction", title='Get your images')
            return filename

        # Gaussian Blur results
        if function_name.get() == 'Gaussian Blur':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./gaussian_blur", title='Get your images')
            return filename

        # Grayscale Histogram results  
        if function_name.get() == 'Grayscale Histogram':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif'))]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./get_grayscale_histogram", title='Get your images data')
            return filename

        # Grayscale results
        if function_name.get() == 'Grayscale':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./grayscaled", title='Get your images')
            return filename

        # Horizontal Concat results 
        if function_name.get() == 'Horizontal Concat':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./h_concatenated", title='Get your images')
            return filename

        # Histogram Equalization results 
        if function_name.get() == 'Histogram Equalization':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif'))]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./get_histogram_equalization", title='Get your images data')
            return filename

        # HSV results 
        if function_name.get() == 'HSV':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./hsv", title='Get your images')
            return filename

        # Luminosity Add results 
        if function_name.get() == 'Luminosity Add':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./luminosity_added", title='Get your images')
            return filename

        # Luminosity Subtract results  
        if function_name.get() == 'Luminosity Subtract':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./luminosity_subtracted", title='Get your images')
            return filename

        # Median Blur results 
        if function_name.get() == 'Median Blur':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./median_blured", title='Get your images')
            return filename

        # Merging RGB results  
        if function_name.get() == 'Merging RGB':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./RGB_merged", title='Get your images')
            return filename

        # Negative results
        if function_name.get() == 'Negative':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./negative", title='Get your images')
            return filename

        # Normalization results  
        if function_name.get() == 'Normalization':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./normalized", title='Get your images')
            return filename

        # Number Of Contours results  
        if function_name.get() == 'Number Of Contours':
            image_formats= [('TXT', ('*.txt'))]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./get_number_contours", title='Get your images data')
            return filename

        # Pixel Values results  
        if function_name.get() == 'Pixel Values':
            image_formats= [('TXT', ('*.txt'))]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./get_pixels", title='Get your images data')
            return filename

        # Rectangle Mask results 
        if function_name.get() == 'Rectangle Mask':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./rectangle_mask", title='Get your images')
            return filename

        # Resize results
        if function_name.get() == 'Resize':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./resized", title='Get your images')
            return filename

         # Rotate 45 Degrees results
        if function_name.get() == 'Rotate 45 Degrees':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./rotated45", title='Get your images')
            return filename
            
        # Rotate 90 Degrees results 
        if function_name.get() == 'Rotate 90 Degrees':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./rotated90", title='Get your images')
            return filename

        # Rotate 180 Degrees results  
        if function_name.get() == 'Rotate 180 Degrees':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./rotated180", title='Get your images')
            return filename

        # Rotate 270 Degrees results 
        if function_name.get() == 'Rotate 270 Degrees':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./rotated270", title='Get your images')
            return filename

        # Shape results
        if function_name.get() == 'Shape':
            image_formats= [('TXT', ('*.txt'))]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./get_shape", title='Get your images data')
            return filename

        # Shift results 
        if function_name.get() == 'Shift':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./shifted", title='Get your images')
            return filename

        # Splitting RGB results  
        if function_name.get() == 'Splitting RGB':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./RGB_splitted", title='Get your images')
            return filename

        # Vertical Concat results
        if function_name.get() == 'Vertical Concat':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./v_concatenated", title='Get your images')
            return filename

        # Vertical And Horizontal Concat results 
        if function_name.get() == 'Vertical And Horizontal Concat':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./vh_concatenated", title='Get your images')
            return filename

        # Zooming results 
        if function_name.get() == 'Zooming':
            image_formats= [('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')]
            filename = askopenfilenames(filetypes=image_formats, initialdir="./zooming", title='Get your images')
            return filename


if __name__ == "__main__":

    # define a function for vertically 
    # concatenating images of the 
    # same size  and horizontally
    def concat_vh(list_2d):        
        # return final image
        return cv2.vconcat([cv2.hconcat(list_h) 
                            for list_h in list_2d])


    # define a function for vertically 
    # concatenating images of different
    # widths 
    def vconcat_resize(img_list, interpolation = cv2.INTER_CUBIC):
        # take minimum width
        w_min = min(img.shape[1] for img in img_list)        
        # resizing images
        im_list_resize = [cv2.resize(img,
                        (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                    interpolation = interpolation)
                        for img in img_list]
        # return final image
        return cv2.vconcat(im_list_resize)



    # negative of image
    def negate(img):        
        return cv2.bitwise_not(img)


    # define a function for horizontally 
    # concatenating images of different
    # heights 
    def hconcat_resize(img_list, 
                    interpolation 
                    = cv2.INTER_CUBIC):
        # take minimum hights
        h_min = min(img.shape[0] 
                    for img in img_list)        
        # image resizing 
        im_list_resize = [cv2.resize(img,
                        (int(img.shape[1] * h_min / img.shape[0]),
                            h_min), interpolation
                                    = interpolation) 
                        for img in img_list]        
        # return final image
        return cv2.hconcat(im_list_resize)


    # Gussian blur using linear separable property of Gaussian distribution
    def fast_gaussian_blur(img, ksize, sigma):    
        kernel_1d = cv2.getGaussianKernel(ksize, sigma)
        return cv2.sepFilter2D(img, -1, kernel_1d, kernel_1d)


    # Gamma correction of image
    def gamma(img, gamma_value):        
        i_gamma = 1 / gamma_value
        lut = np.array([((i / 255) ** i_gamma) * 255 for i in np.arange(0, 256)], dtype = 'uint8')
        return cv2.LUT(img, lut)


    # return list of all subsets of length r
    # to deal with duplicate subsets use 
    # set(list(combinations(arr, r)))
    def rSubset(arr, r):
        return list(combinations(arr, r))


    # Blacking and Whiting out indices same as color balance
    def get_black_white_indices(hist, tot_count, black_count, white_count):
        black_ind = 0
        white_ind = 255
        co = 0
        for i in range(len(hist)):
            co += hist[i]
            if co > black_count:
                black_ind = i
                break                
        co = 0
        for i in range(len(hist) - 1, -1, -1):
            co += hist[i]
            if co > (tot_count - white_count):
                white_ind = i
                break        
        return [black_ind, white_ind]


    # Contrast stretch image with black and white cap
    def contrast_stretch(img, black_point, white_point):    
        tot_count = img.shape[0] * img.shape[1]
        black_count = tot_count * black_point / 100
        white_count= tot_count * white_point / 100
        ch_hists = []
        # calculate histogram for each channel
        for ch in cv2.split(img):
            ch_hists.append(cv2.calcHist([ch], [0], None, [256], (0, 256)).flatten().tolist())        
        # get black and white percentage indices
        black_white_indices = []
        for hist in ch_hists:
            black_white_indices.append(get_black_white_indices(hist, tot_count, black_count, white_count))            
        stretch_map = np.zeros((3, 256), dtype = 'uint8')        
        # Stretch histogram 
        for curr_ch in range(len(black_white_indices)):
            black_ind, white_ind = black_white_indices[curr_ch]
            for i in range(stretch_map.shape[1]):
                if i < black_ind:
                    stretch_map[curr_ch][i] = 0
                else:
                    if i > white_ind:
                        stretch_map[curr_ch][i] = 255
                    else:
                        if (white_ind - black_ind) > 0:
                            stretch_map[curr_ch][i] = round((i - black_ind) / (white_ind - black_ind)) * 255
                        else:
                            stretch_map[curr_ch][i] = 0        
        # Stretch image
        ch_stretch = []
        for i, ch in enumerate(cv2.split(img)):
            ch_stretch.append(cv2.LUT(ch, stretch_map[i]))            
        return cv2.merge(ch_stretch)


    # Normalize kernel
    def normalize_kernel(kernel, k_width, k_height, scaling_factor = 1.0):
        '''Zero-summing normalize kernel'''    
        K_EPS = 1.0e-12
        # positive and negative sum of kernel values
        pos_range, neg_range = 0, 0
        for i in range(k_width * k_height):
            if abs(kernel[i]) < K_EPS:
                kernel[i] = 0.0
            if kernel[i] < 0:
                neg_range += kernel[i]
            else:
                pos_range += kernel[i]        
        # scaling factor for positive and negative range
        pos_scale, neg_scale = pos_range, -neg_range
        if abs(pos_range) >= K_EPS:
            pos_scale = pos_range
        else:
            pos_sacle = 1.0
        if abs(neg_range) >= K_EPS:
            neg_scale = 1.0
        else:
            neg_scale = -neg_range
            
        pos_scale = scaling_factor / pos_scale
        neg_scale = scaling_factor / neg_scale        
        # scale kernel values for zero-summing kernel
        for i in range(k_width * k_height):
            if (not np.nan == kernel[i]):
                kernel[i] *= pos_scale if kernel[i] >= 0 else neg_scale                
        return kernel


    # Difference of Gaussian
    def dog(img, k_size, sigma_1, sigma_2):
        '''Difference of Gaussian by subtracting kernel 1 and kernel 2'''        
        k_width = k_height = k_size
        x = y = (k_width - 1) // 2
        kernel = np.zeros(k_width * k_height)        
        # first gaussian kernal
        if sigma_1 > 0:
            co_1 = 1 / (2 * sigma_1 * sigma_1)
            co_2 = 1 / (2 * np.pi * sigma_1 * sigma_1)
            i = 0
            for v in range(-y, y + 1):
                for u in range(-x, x + 1):
                    kernel[i] = np.exp(-(u*u + v*v) * co_1) * co_2
                    i += 1
        # unity kernel
        else:
            kernel[x + y * k_width] = 1.0        
        # subtract second gaussian from kernel
        if sigma_2 > 0:
            co_1 = 1 / (2 * sigma_2 * sigma_2)
            co_2 = 1 / (2 * np.pi * sigma_2 * sigma_2)
            i = 0
            for v in range(-y, y + 1):
                for u in range(-x, x + 1):
                    kernel[i] -= np.exp(-(u*u + v*v) * co_1) * co_2
                    i += 1
        # unity kernel
        else:
            kernel[x + y * k_width] -= 1.0        
        # zero-normalize scling kernel with scaling factor 1.0
        norm_kernel = normalize_kernel(kernel, k_width, k_height, scaling_factor = 1.0)        
        # apply filter with norm_kernel
        return cv2.filter2D(img, -1, norm_kernel.reshape(k_width, k_height))

    app = App()
    app.config(menu=app.menubar)

    button_frame = tk.Frame(app)
    button_frame.pack(fill=tk.X, side=tk.TOP)

    # Combobox OpenCV Actions    
    function_name = tk.StringVar()
    option = ttk.Combobox(button_frame, textvariable=function_name)
    option['values'] = ["Bitwise AND",
    "Bitwise NOT",
    "Bitwise OR",
    "Bitwise XOR",
    "Blending",
    "Blur",
    "Box Filter",
    "Circle Mask",
    "Color Histogram",
    "Contours",
    "Contrast Stretching",
    "Crop",
    "Dilation",
    "Dog",
    "Drawing Circle",
    "Drawing Image",
    "Drawing Rectangle",
    "Drawing Text",
    "DType",
    "Edge Cascade",
    "Enhancement",
    "Erosion",
    "Flipping",
    "Gamma Correction",
    "Gaussian Blur",
    "Grayscale Histogram",
    "Grayscale",
    "Horizontal Concat",
    "Histogram Equalization",
    "HSV",
    "Luminosity Add",
    "Luminosity Subtract",
    "Median Blur",
    "Merging RGB",
    "Negative",
    "Normalization",
    "Number Of Contours",
    "Pixel Values",
    "Rectangle Mask",
    "Resize",
    "Rotate 45 Degrees",
    "Rotate 90 Degrees",
    "Rotate 180 Degrees",
    "Rotate 270 Degrees",
    "Shape",
    "Shift",
    "Splitting RGB",
    "Vertical Concat",
    "Vertical And Horizontal Concat",
    "Zooming"]
    option['state'] = 'readonly'    

    open_button = tk.Button(button_frame, text='Open', bg='#eee', height='2', command=app.openImage)
    clear_button = tk.Button(button_frame, text='Clear', bg='#eee', height='2', command=app.clear)
    preview_button = tk.Button(button_frame, text='Preview', bg='#e5e5ff', height='2', command=app.preview)
    save_button = tk.Button(button_frame, text='Save', bg='#e5ffe5', height='2', command=app.save)
    results_button = tk.Button(button_frame, text='Results', bg='#ffffe5', height='2', command=app.results)
    exit_button = tk.Button(button_frame, text='Exit', bg='#eee', height='2', command=lambda: app.quit())
    
    button_frame.columnconfigure(0, weight=1)
    button_frame.columnconfigure(1, weight=1)
    button_frame.columnconfigure(2, weight=1)
    button_frame.columnconfigure(3, weight=1)
    button_frame.columnconfigure(4, weight=1)
    button_frame.columnconfigure(5, weight=1)
    button_frame.columnconfigure(6, weight=1)

    option.grid(row=0, column=0, ipady=10, sticky=tk.W+tk.E)
    open_button.grid(row=0, column=1, sticky=tk.W+tk.E)
    clear_button.grid(row=0, column=2, sticky=tk.W+tk.E)
    preview_button.grid(row=0, column=3, sticky=tk.W+tk.E)
    save_button.grid(row=0, column=4, sticky=tk.W+tk.E)
    results_button.grid(row=0, column=5, sticky=tk.W+tk.E)
    exit_button.grid(row=0, column=6, sticky=tk.W+tk.E)    

    scrollbar = Scrollbar(app, orient="vertical")
    
    scrollbar.pack(side="right", fill="y")

    # create listbox
    listBox = Listbox(app)
    listBox.pack(fill=tk.BOTH, expand=True)

    scrollbar.config(command=listBox.yview)
    listBox.config(yscrollcommand=scrollbar.set)

    app.mainloop()
