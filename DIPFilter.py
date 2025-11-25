#%%
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2


class DIPFilters:
    def __init__(self, image_path):
        
        # Read image into Python
        self.image_path = image_path
        self.img = mpimg.imread(self.image_path)
        
        # Format for OpenCV Canny edge
        self.img_filtered = (self.img*255).astype(np.uint8)
    
    def display(self):
        
        # Display the unfiltered image
        plt.figure()
        plt.title(f"Unfiltered")
        plt.imshow(self.img, cmap="gray")
    
    def col2greyscale(self):
        # Convert colour images to greyscale
        # For testing larger images from the internet
        if self.img.ndim == 2:
            print("Image is already greyscale.")
        else:
            rgb_weights = np.array([0.299, 0.587, 0.114])
            self.img = np.dot(self.img[..., :3], rgb_weights)
    
    def crudeMeanFilter(self, window_size):
        
        # crude mean filter, not separable
        # Designed for comparison with SeparableMeanFilter
        
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
        
        # Pad the image with the edge pixel
        # Gives room for the window at position 1 
        padded_img = np.pad(self.img, pad_width=size, mode='edge')
        
        img_height = self.img.shape[0]
        img_width = self.img.shape[1]
        
        # Create an empty array to store the image
        # After the passing of the horizontal mean filter
        intermediateArray = np.zeros_like(padded_img)
        
        # Horizontal filter pass
        for row_index in range(size, img_height+size):
            for pixel_index in range(size, img_width+size):
                # Find pixel window
                window = padded_img[row_index, (pixel_index-size):(pixel_index+size+1)]
                # Find the mean of all pixels in window
                total = np.sum(window)
                new_pixel = total/window_area
                intermediateArray[row_index, pixel_index] = new_pixel
            
        # Create an empty array to store the final filtered image
        blurredImg = np.zeros_like(intermediateArray)
        
        # Vertical filter pass
        for row_index in range(size, img_height+size):
            for pixel_index in range(size, img_width+size):
                # Find pixel window
                window = intermediateArray[row_index-size:row_index+size+1, pixel_index]
                # Find mean of all pixels in window
                total = np.sum(window)
                new_pixel = total/window_area
                blurredImg[row_index, pixel_index] = new_pixel 
        
        # Remove padding 
        slicedArray = blurredImg[size:img_height+size, size:img_width+size]
        self.img_filtered = (slicedArray*255).astype(np.uint8)
        
        # Show the filtered Image
        plt.figure()
        plt.title(f"Separable Mean Filter {window_size}x{window_size}")
        plt.imshow(slicedArray, cmap="gray")  

    def GaussianFilter(self,  sd):
        # Window size calculated from standard deviation
        # = 2 * ceil(3*sd) + 1
        
        size = math.ceil(3*sd)
        window_size = 2*size + 1
        window_area = window_size*window_size
        
        gauss = lambda x : math.exp(-((x**2)/(2*sd**2)))
        # Create horizontal window of gaussian filter
        h_window = np.array([gauss(x) for x in range(-size, size+1)])
        # Create vertical window of gaussian filter
        v_window = np.transpose(h_window)

        # Pad the edges
        padded_img = np.pad(self.img, pad_width=size, mode='edge')
        
        img_height = self.img.shape[0]
        img_width = self.img.shape[1]
        
        # Create empty array to hold image after horizontal filter pass
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
        
        
        # Remove padding 
        slicedArray = blurredImg[size:img_height+size, size:img_width+size]
        self.img_filtered = (slicedArray*255).astype(np.uint8)
        
        # Show filtered image
        plt.figure()
        plt.title(f"Gaussian Filter, SD = {sd}")
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
        
        # Pad edges of image
        padded_img_disc = np.pad(img_disc, pad_width=size, mode='edge')
        
        img_height = self.img.shape[0]
        img_width = self.img.shape[1]
        
        # Create empty array to hold filtered image
        blurredImg = np.zeros_like(padded_img_disc)
        
        # Create empty histogram
        histogram = [0]*256
        median_idx = (window_area//2)+1
        
        # Main loop through image
        for row_index in range(size, img_height+size):
            
            # Empty histogram each row
            histogram = [0]*256
            pixel_index = size
            # Loop through the window and update the histogram
            for hist_row_index in range(-size, size+1):
                for hist_pixel_index in range(-size, size+1):
                    pixel_val = padded_img_disc[row_index + hist_row_index, pixel_index + hist_pixel_index]
                    histogram[int(pixel_val)] += 1
            
            # Find the median value from histogram just made
            blurredImg[row_index, pixel_index] = findMedian(histogram, median_idx)
            
            # Update histogram for rest of pixels in row
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
            
                # Find median from updated histogram
                blurredImg[row_index, pixel_index] = findMedian(histogram, median_idx)
        
        # Slice off padding        
        slicedArray = blurredImg[size:img_height+size, size:img_width+size]
        self.img_filtered = slicedArray.astype(np.uint8)
        
        # Show the filtered image
        plt.figure()
        plt.title(f"Median Filter {window_size}x{window_size}")
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
            
            # Function for finding the truncated median from a histogram 
            
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
        
        # Pad the image
        padded_img_disc = np.pad(img_disc, pad_width=size, mode='constant')
        
        img_height = self.img.shape[0]
        img_width = self.img.shape[1]
        
        # Create empty array for holding the filtered image
        blurredImg = np.zeros_like(padded_img_disc)
        
        # Create empty histogram
        histogram = [0]*256
        median_idx = (window_area//2)+1
        
        # Main loop through image
        for row_index in range(size, img_height+size):
            
            # Clear histogram for eahc row
            histogram = [0]*256
            pixel_index = size
            # Create an array of all the current pixels in the window
            current_pixels = padded_img_disc[row_index-size:row_index+size+1, pixel_index-size:pixel_index+size+1].flatten()
            
            # Loop through window and Update histogram 
            for hist_row_index in range(-size, size+1):
                for hist_pixel_index in range(-size, size+1):
                    pixel_val = padded_img_disc[row_index + hist_row_index, pixel_index + hist_pixel_index]
                    histogram[int(pixel_val)] += 1
            
            # Find the truncated median value from hist just made
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
            
                # Find truncated median from updated histogram
                blurredImg[row_index, pixel_index] = findTruncMedian(current_pixels, histogram, median_idx)
                
                
        # Slice off padding        
        slicedArray = blurredImg[size:img_height+size, size:img_width+size]
        self.img_filtered = slicedArray.astype(np.uint8)
            
        # Show filtered image
        plt.figure()
        plt.title(f"Truncated Median Filter {window_size}x{window_size}")
        plt.imshow(slicedArray, cmap="gray")

    def CentreWeightedMedian(self, window_size, centre_weight=1):
                
        def findWeightedMedian(hist, median_idx, centre_val, centre_weight):
            # Function for finding the weighted median from a histogram
            counter = 0 
            for pixel in range(256):
                current_bar_size = hist[pixel]
                # Add extra occurences for the central value
                if pixel == centre_val:
                    current_bar_size += centre_weight
                counter += current_bar_size
                if counter >= median_idx:
                    return pixel
            return 0
        
        # Discretise image for histogram sort
        img_disc = np.round(self.img*255)
        
        size = int((window_size-1)/2)
        window_area = window_size*window_size
        
        # Pad the image
        padded_img_disc = np.pad(img_disc, pad_width=size, mode='constant')
        
        img_height = self.img.shape[0]
        img_width = self.img.shape[1]
        
        # Create empty array to hold filtered image
        blurredImg = np.zeros_like(padded_img_disc)
        
        # Create empty histogram
        histogram = [0]*256
        median_idx = (window_area+centre_weight-1)/2
        
        # Main loop through image
        for row_index in range(size, img_height+size):
            
            # Clear histogram at start of each row 
            histogram = [0]*256
            pixel_index = size
            
            # Find the central pixel value of the window
            centre_val = padded_img_disc[row_index, pixel_index]
            
            # Update histogram
            for hist_row_index in range(-size, size+1):
                for hist_pixel_index in range(-size, size+1):
                    pixel_val = padded_img_disc[row_index + hist_row_index, pixel_index + hist_pixel_index]
                    histogram[int(pixel_val)] += 1
            
            # Find the weighted median value from hist just made
            blurredImg[row_index, pixel_index] = findWeightedMedian(histogram, median_idx, centre_val, centre_weight)
            
            # Update histrogram for rest of pixels in row
            for pixel_index in range(size+1, img_width+size):
                
                old_col = pixel_index - size - 1
                new_col = pixel_index + size
                
                centre_val = padded_img_disc[row_index, pixel_index]
                
                for col_row_index in range(-size, size+1):
                    
                    # Remove pixel values from hist from column that just left
                    old_pixel_val = padded_img_disc[row_index + col_row_index, old_col]
                    histogram[int(old_pixel_val)] -= 1
                    
                    # Add pixel values to hist from column that just joined
                    new_pixel_val = padded_img_disc[row_index + col_row_index, new_col]
                    histogram[int(new_pixel_val)] += 1
            
                # Find median from updated hisogram
                blurredImg[row_index, pixel_index] = findWeightedMedian(histogram, median_idx, centre_val, centre_weight)
        
        # Slice off padding        
        slicedArray = blurredImg[size:img_height+size, size:img_width+size]
        self.img_filtered = slicedArray.astype(np.uint8)
            
        plt.figure()
        plt.title(f"Centre Weighted Median Filter {window_size}x{window_size}")
        plt.imshow(slicedArray, cmap="gray")  
        pass
        
    def OpenClose(self, window_size):
        
        def erosion(img, window_size):
            size = int((window_size-1)/2)
            img_height = img.shape[0]
            img_width = img.shape[1]
            
            padded_img = np.pad(img, pad_width=size, mode='edge')
            
            erodedImg = np.zeros_like(padded_img)  

            for row_index in range(size, img_height+size):
                for pixel_index in range(size, img_width+size):
                    window = padded_img[row_index-size:row_index+size+1, pixel_index-size:pixel_index+size+1]
                    erodedImg[row_index, pixel_index] = np.min(window)
              
            return erodedImg[size:img_height+size, size:img_width+size]
        
        def dilation(img, window_size):
            size = int((window_size-1)/2)
            img_height = img.shape[0] 
            img_width = img.shape[1] 
            
            padded_img = np.pad(img, pad_width=size, mode='edge')
        
            dilatedImg = np.zeros_like(padded_img)

            for row_index in range(size, img_height+size):
                for pixel_index in range(size, img_width+size):
                    window = padded_img[row_index-size:row_index+size+1, pixel_index-size:pixel_index+size+1]
                    dilatedImg[row_index, pixel_index] = np.max(window)
              
            return dilatedImg[size:img_height+size, size:img_width+size]

        # Opening
        step1_eroded = erosion(self.img, window_size)
        step2_opened = dilation(step1_eroded, window_size)
        
        # Closing
        step3_dilated = dilation(step2_opened, window_size)
        final_open_closed = erosion(step3_dilated, window_size)
        
        self.img_filtered = (final_open_closed*255).astype(np.uint8)
            
        plt.figure()
        plt.title(f"Open-Close Filter {window_size}x{window_size}")
        plt.imshow(final_open_closed, cmap="gray") 
                          
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
    # NZfilter.CannyEdgeDetect(150,300)
    # NZfilter.crudeMeanFilter(7)
    NZfilter.SeparableMeanFilter(5)
    NZfilter.GaussianFilter(4)
    NZfilter.MedianFilter(5)
    NZfilter.TruncatedMedian(5)
    NZfilter.CentreWeightedMedian(7,7)
    NZfilter.OpenClose(3)


    # CarWindow = DIPFilters('Images/carwindow.jpg')
    # CarWindow.display()
    # CarWindow.col2greyscale()
    # CarWindow.display()
    # CarWindow.SeparableMeanFilter(150)







 # %%
