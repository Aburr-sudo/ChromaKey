#!/usr/bin/env python
# coding: utf-8

# In[2]:


##### CSCI935 Assignment 1 ####
## Allan Burr
## Student Number: 4989272

##Import Libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys # to parse command line arguments
import matplotlib.image as mpimg


# In[3]:


def isolateChannels(image):
    channel_1 = image[:,:,0]
    channel_2 = image[:,:,1]
    channel_3 = image[:,:,2]
    
    return channel_1, channel_2, channel_3


# In[10]:


def convertToColorSpace(image, color_space):
    converted = image.copy()
    if color_space == "-XYZ":
        converted = cv2.cvtColor(converted, cv2.COLOR_BGR2XYZ)
        print("XYZ CONVERSION PROCEEDING")
        converted = converted/255.0
        return converted
    elif color_space == "-Lab":
        converted = cv2.cvtColor(converted, cv2.COLOR_BGR2Lab)
        return converted
    elif color_space == "-YCrCb":
        converted = cv2.cvtColor(converted, cv2.COLOR_BGR2YCR_CB)
        return converted
    elif color_space == "-HSV":
        converted = cv2.cvtColor(converted, cv2.COLOR_BGR2HSV)
        return converted
    else:
        print("pls input a valid colorspace")


# In[5]:


def displayImagesTask1(image1, image2, image3, image4):
    
    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.subplot(2, 2, 1).set_title("Original Image")
    plt.imshow(image1)
    
    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.imshow(image2, cmap='gray')
    
    plt.subplot(2,2,3)
    plt.axis("off")
    plt.imshow(image3, cmap='gray')
    
    plt.subplot(2,2,4)
    plt.axis("off")
    plt.imshow(image4, cmap='gray')
    
    for i in range(1,4):
        plt.subplot(2,2,i+1).set_title('Channel number {}' .format(i))
    plt.show()


# In[ ]:


def scaleImages(image1, image2, image3):
    image1 = image1/255.0
    image2 = image2/255.0
    image3 = image3/255.0
    return image1,image2,image3 


# In[ ]:


def displayImagesTask2(image1, image2, image3, image4):
    plt.subplot(2, 2, 1)
    plt.axis("off")
    plt.imshow(image1)
    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.imshow(image2)
    plt.subplot(2,2,3)
    plt.axis("off")
    plt.imshow(image3)
    plt.subplot(2,2,4)
    plt.axis("off")
    plt.imshow(image4)
    plt.show()


# In[ ]:


"""
To display an image using openCV:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',basic_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""


# In[6]:


def task1(color_space, image_name):
    image = cv2.imread(image_name)
    rgb_corrected = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    converted = convertToColorSpace(rgb_corrected, color_space)
    channel_1, channel_2, channel_3 = isolateChannels(converted) 
    channel_1, channel_2, channel_3 = scaleImages(channel_1, channel_2, channel_3)
    displayImagesTask1(rgb_corrected, channel_1, channel_2, channel_3)

    ## Regulate size of images in a single window ##
   


# In[8]:


def task2(image1, image2):
    basic_image = cv2.imread(firstImage)
    background = cv2.imread(secondImage)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    person_image = cv2.cvtColor(basic_image, cv2.COLOR_BGR2RGB)
    height, width = getDimensions(basic_image)
    background_resized = cv2.resize(background, (width, height))
    
    #default values
    ##    lower_green = np.array([0,180,0])
    ##    upper_green = np.array([100,255,100])
    # the first and last greenscreen images are the most difficult as the background is not of a uniform intensity
    lower_green = np.array([0,120,0])
    upper_green = np.array([140,255,140])
    mask = cv2.inRange(person_image, lower_green, upper_green)
    
    masked_image = np.copy(person_image)
    masked_image[mask != 0] = 255
    
    back_image = background_resized.copy()
    #hollows out a stencil in the background image to fit the foreground image into
    back_image[mask == 0] = 0
    # Adds the extracted person into the stencil image, matrix addition facilitated by numpy
    final_image = back_image + masked_image
    
    displayImagesTask2(person_image, masked_image, background_resized, final_image)
    


# In[8]:


def getDimensions(basic_image):
    height = basic_image.shape[0]
    width = basic_image.shape[1]
    return height, width


# In[ ]:





# In[9]:


if __name__ == "__main__":
    numArgs = len(sys.argv)
    if numArgs > 3:
        print("Please enter an acceptable amount of arguments")
        exit()
        
    firstArgument = sys.argv[1]
    secondArgument = sys.argv[2]

    if firstArgument[0] == '-':
        colour_space = firstArgument
        image_name = secondArgument
        print('Performing task 1')
        task1(colour_space, image_name)

    else:
        firstImage = firstArgument
        secondImage = secondArgument
        print('Performing task 2')
        task2(firstImage, secondImage)


# In[4]:





# In[5]:





# In[6]:





# In[1]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:




#xyz = img.copy()
#xyz = cv2.cvtColor(xyz, cv2.COLOR_BGR2RGB)
#b,g,r = cv2.split(img)
#reconstructed = cv2.merge((b,g,r))
#cv2.imshow('image',xyz)
#cv2.imshow('image',reconstructed)
# B = 0, G = 1, R = 2
#red_channel = img[:,:,2]
#red_img = np.zeros(img.shape)
#assign the red channel of src to empty image
#red_img[:,:,2] = red_channel


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




