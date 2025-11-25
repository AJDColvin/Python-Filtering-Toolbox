#%%
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2


class DIPFilters:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = mpimg.imread(self.image_path)
        
        # Format for OpenCV Canny edge
        self.img_filtered = (self.img*255).astype(np.uint8)
    
    def display(self):
        plt.figure()
        plt.title("Unfiltered")
        plt.imshow(self.img, cmap="gray")
        self.img_filtered = (self.img*255).astype(np.uint8)
    
    def col2greyscale(self):
        if self.img.ndim == 2:
            print("Image is already greyscale.")
        else:
            rgb_weights = np.array([0.299, 0.587, 0.114])
            self.img = np.dot(self.img[..., :3], rgb_weights)
    
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
        
    def SeparableMeanFilter(self, window_size):
        
        #TODO: deal width the averaging of 0s in the edge
        
        size = int((window_size-1)/2)
        window_area = window_size*window_size
        padded_img = np.pad(self.img, pad_width=size, mode='constant')
        
        img_height = self.img.shape[0]
        img_width = self.img.shape[1]
        
        intermediateArray = np.zeros_like(padded_img)
        
        # Horizontal filter pass
        for row_index in range(size, img_height+size):
            # print("START OF ROW")
            for pixel_index in range(size, img_width+size):
                window = padded_img[row_index, (pixel_index-size):(pixel_index+size+1)]
                # print('Window', window)
                total = np.sum(window)
                new_pixel = total/window_area
                intermediateArray[row_index, pixel_index] = new_pixel
            
        blurredImg = np.zeros_like(intermediateArray)
        
        # Vertical filter pass
        for row_index in range(size, img_height+size):
            for pixel_index in range(size, img_width+size):
                window = intermediateArray[row_index-size:row_index+size+1, pixel_index]
                total = np.sum(window)
                new_pixel = total/window_area
                blurredImg[row_index, pixel_index] = new_pixel 
        
        
        slicedArray = blurredImg[size:img_height+size, size:img_width+size]
        self.img_filtered = (slicedArray*255).astype(np.uint8)
        
        plt.figure()
        plt.title("Separable Mean Filter")
        plt.imshow(slicedArray, cmap="gray")  

    def GaussianFilter(self,  sd):
        # Window size calculated from standard deviation
        # = 2 * ceil(3*sd) + 1
        
        size = math.ceil(3*sd)
        window_size = 2*size + 1
        window_area = window_size*window_size
        
        gauss = lambda x : math.exp(-((x**2)/(2*sd**2)))
        h_window = np.array([gauss(x) for x in range(-size, size+1)])
        v_window = np.transpose(h_window)

        padded_img = np.pad(self.img, pad_width=size, mode='constant')
        
        img_height = self.img.shape[0]
        img_width = self.img.shape[1]
        
        intermediateArray = np.zeros_like(padded_img)
        
        # Horizontal filter pass
        for row_index in range(size, img_height+size):
            # print("START OF ROW")
            for pixel_index in range(size, img_width+size):
                window = padded_img[row_index, (pixel_index-size):(pixel_index+size+1)]
                total = np.dot(window, h_window)
                new_pixel = total/window_area
                intermediateArray[row_index, pixel_index] = new_pixel
            
        blurredImg = np.zeros_like(intermediateArray)
        
        # Vertical filter pass
        for row_index in range(size, img_height+size):
            for pixel_index in range(size, img_width+size):
                window = intermediateArray[row_index-size:row_index+size+1, pixel_index]
                total = np.dot(window, v_window)
                new_pixel = total/window_area
                blurredImg[row_index, pixel_index] = new_pixel 
        
        
        slicedArray = blurredImg[size:img_height+size, size:img_width+size]
        self.img_filtered = (slicedArray*255).astype(np.uint8)
        
        plt.figure()
        plt.title("Gaussian Filter")
        plt.imshow(slicedArray, cmap="gray")

    def MedianFilter(self, window_size):
        
        def findMedian(hist, median_idx):
            # Function for finding the median from a histogram
            counter = 0 
            for pixel in range(256):
                counter += hist[pixel]
                if counter >= median_idx:
                    return pixel
            return 0
        
        # Discretise image for histogram sort
        img_disc = np.round(self.img*255)
        
        size = int((window_size-1)/2)
        window_area = window_size*window_size
        padded_img_disc = np.pad(img_disc, pad_width=size, mode='constant')
        
        img_height = self.img.shape[0]
        img_width = self.img.shape[1]
        
        blurredImg = np.zeros_like(padded_img_disc)
        
        histogram = [0]*256
        median_idx = (window_area//2)+1
        
        # Main loop through image
        for row_index in range(size, img_height+size):
            
            # Initialise histogram for row
            histogram = [0]*256
            pixel_index = size
            for hist_row_index in range(-size, size+1):
                for hist_pixel_index in range(-size, size+1):
                    pixel_val = padded_img_disc[row_index + hist_row_index, pixel_index + hist_pixel_index]
                    histogram[int(pixel_val)] += 1
            
            # Find the median value from hist just made
            blurredImg[row_index, pixel_index] = findMedian(histogram, median_idx)
            
            # Update histrogram for rest of pixels in row
            for pixel_index in range(size+1, img_width+size):
                old_col = pixel_index - size - 1
                new_col = pixel_index + size
                
                for col_row_index in range(-size, size+1):
                    
                    # Remove pixel values from hist from column that just left
                    old_pixel_val = padded_img_disc[row_index + col_row_index, old_col]
                    histogram[int(old_pixel_val)] -= 1
                    
                    # Add pixel values to hist from column that just joined
                    new_pixel_val = padded_img_disc[row_index + col_row_index, new_col]
                    histogram[int(new_pixel_val)] += 1
            
                # Find median from updated hisogram
                blurredImg[row_index, pixel_index] = findMedian(histogram, median_idx)
        
        # Slice off padding        
        slicedArray = blurredImg[size:img_height+size, size:img_width+size]
        self.img_filtered = slicedArray.astype(np.uint8)
            
        plt.figure()
        plt.title("Median Filter")
        plt.imshow(slicedArray, cmap="gray")          
 
    def TruncatedMedian(self, window_size):
        
        def findMedian(hist, median_idx):
            # Function for finding the median from a histogram
            counter = 0 
            for pixel in range(256):
                counter += hist[pixel]
                if counter >= median_idx:
                    return pixel
            return 0
        
        def findTruncMedian(current_pixels, hist, median_idx):
            
            # Find the original median of pixel window
            orig_median = findMedian(hist, median_idx)
            
            # Set truncated histogram and median pos = old histogram and median pos
            trunc_hist = hist.copy()
            trunc_median_idx = median_idx
            
            # Find difference between median and left and right most val
            left_range = abs(orig_median - np.min(current_pixels))
            right_range = abs(orig_median - np.max(current_pixels))
            
            # If the right range is larger
            if left_range <= right_range:
                for pixel in current_pixels:
                    # If pixel is out of truncated range, 
                    if pixel > (orig_median + left_range):
                        trunc_hist[int(pixel)] -= 1 
                        trunc_median_idx -= 0.5 # Removing 2 values moves median idx left 1
            
                return findMedian(trunc_hist, math.ceil(trunc_median_idx))
            
            # If the left range is larger
            else:
                for pixel in current_pixels:
                    if pixel < orig_median - right_range:
                        trunc_hist[int(pixel)] -= 1
                        trunc_median_idx -= 0.5
            
                return findMedian(trunc_hist, math.ceil(trunc_median_idx))
                
        # Discretise image for histogram sort
        img_disc = np.round(self.img*255)
        
        size = int((window_size-1)/2)
        window_area = window_size*window_size
        padded_img_disc = np.pad(img_disc, pad_width=size, mode='constant')
        
        img_height = self.img.shape[0]
        img_width = self.img.shape[1]
        
        blurredImg = np.zeros_like(padded_img_disc)
        
        histogram = [0]*256
        median_idx = (window_area//2)+1
        
        # Main loop through image
        for row_index in range(size, img_height+size):
            
            # Initialise histogram for row
        
            histogram = [0]*256
            pixel_index = size
            current_pixels = padded_img_disc[row_index-size:row_index+size+1, pixel_index-size:pixel_index+size+1].flatten()
            
            for hist_row_index in range(-size, size+1):
                for hist_pixel_index in range(-size, size+1):
                    pixel_val = padded_img_disc[row_index + hist_row_index, pixel_index + hist_pixel_index]
                    histogram[int(pixel_val)] += 1
            
            # Find the median value from hist just made
            blurredImg[row_index, pixel_index] = findTruncMedian(current_pixels, histogram, median_idx)
            
            
            # Update histrogram for rest of pixels in row
            for pixel_index in range(size+1, img_width+size):
                
                current_pixels = padded_img_disc[row_index-size:row_index+size+1, pixel_index-size:pixel_index+size+1].flatten()
                old_col = pixel_index - size - 1
                new_col = pixel_index + size
                
                for col_row_index in range(-size, size+1):
                    
                    # Remove pixel values from hist from column that just left
                    old_pixel_val = padded_img_disc[row_index + col_row_index, old_col]
                    histogram[int(old_pixel_val)] -= 1
                    
                    # Add pixel values to hist from column that just joined
                    new_pixel_val = padded_img_disc[row_index + col_row_index, new_col]
                    histogram[int(new_pixel_val)] += 1
            
                # Find median from updated histogram
                blurredImg[row_index, pixel_index] = findTruncMedian(current_pixels, histogram, median_idx)
                
                
        # Slice off padding        
        slicedArray = blurredImg[size:img_height+size, size:img_width+size]
        self.img_filtered = slicedArray.astype(np.uint8)
            
        plt.figure()
        plt.title("Truncated Median Filter")
        plt.imshow(slicedArray, cmap="gray")
        
        
                
    def CannyEdgeDetect(self, lower, upper):
        # Uses OpenCV's Canny Edge detector
        # To verify the performance of the filtered images
        
        edges = cv2.Canny(self.img_filtered, lower, upper)
        
        plt.figure()
        plt.title("Canny Edge Detection")
        plt.imshow(edges, cmap="gray")

            
           
        
        
        
        


        

#%%
if __name__ == '__main__':
    NZfilter = DIPFilters('Images/NZjers1.png')
    NZfilter.display()
    NZfilter.CannyEdgeDetect(150,300)
    # NZfilter.crudeMeanFilter(7)
    NZfilter.SeparableMeanFilter(5)
    NZfilter.GaussianFilter(4)
    NZfilter.MedianFilter(3)
    NZfilter.TruncatedMedian(5)
    NZfilter.CannyEdgeDetect(150,300)


    # CarWindow = DIPFilters('Images/carwindow.jpg')
    # CarWindow.display()
    # CarWindow.col2greyscale()
    # CarWindow.display()
    # CarWindow.SeparableMeanFilter(150)







 # %%
