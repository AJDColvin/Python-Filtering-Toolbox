#%%
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


class DIPFilters:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = mpimg.imread(self.image_path)
    
    def display(self):
        plt.figure()
        plt.imshow(self.img, cmap="gray")
    
    def crudeMeanFilter(self, window_size):
        
        window_size = window_size
        size = int((window_size-1)/2)

        plt.figure()

        # Empty array of height and width of the image
        img_height = self.img.shape[0]
        img_width = self.img.shape[1]
        emptyArray = np.zeros((img_height, img_width))

        np.pad(emptyArray, pad_width=size, mode='constant')

        for row_index in range(size,len(self.img)-size):
            for pixel_index in range(size,len(self.img[row_index])-size):
                window = [row[pixel_index-size:pixel_index+size+1] for row in self.img[row_index-size:row_index+size+1]]
                total = np.sum(window)
                length = np.array(window).size
                new_pixel = total/length
                emptyArray[row_index, pixel_index] = new_pixel

        slicedArray = emptyArray[size:img_height-size, size:img_width-size]
        plt.imshow(slicedArray, cmap="gray")
    
    def EfficientMeanFilter(self, window_size):
        
        size = int((window_size-1)/2)
        window_area = window_size*window_size
        
        padded_img = np.pad(self.img, pad_width=size, mode='constant')
        
        img_height = self.img.shape[0]
        img_width = self.img.shape[1]
        blurredImg = np.zeros((img_height, img_width))
        
        for row_index in range(img_height):
            for pixel_index in range(img_width):
                window = padded_img[row_index: row_index+window_size, pixel_index: pixel_index+window_size]
                total = np.sum(window)
                new_pixel = total/window_area
                blurredImg[row_index, pixel_index] = new_pixel
        
        plt.figure()
        plt.imshow(blurredImg, cmap="gray")
                       
        
        
    

    def GaussianFilter(self, window_size):
        pass


        

    

NZfilter = DIPFilters('Images/NZjers1.png')
NZfilter.display()
NZfilter.crudeMeanFilter(7)
NZfilter.EfficientMeanFilter(7)


