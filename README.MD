This python script has two functions:

1. It can take an image and convert it into another color space (XYZ, Lab, HSV or YCbCr)
It will then display the original image alongside each separated channel of the same image in the specified color space
to run this function input the following

python chromaKey.py [-Color Space] [image]

2. It can take two images, one being a green screen image, and extract the foreground from this and superimpose it onto the second image
To run this function input the following:

python chromaKey.py [green screen image] [background image]

To see an example of the second functionality see Example.jpeg
