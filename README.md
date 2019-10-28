# Scan Converter

In this repository, I develop a small program to solve an everyday task of mine - to crop and transform a sheet of paper from a photo. The subtasks are:
 - find the corners of the sheet of paper in the image
 - transform the image with its perspective
 - downscale the image
 - save as small PDF file

The program can be run as script. If it is invoked without arguments, it takes the current working directory as image path to work on. Otherwise, the image path to work on can be specified as argument.

For any `*.jpg` or `*.jpg` files, it calculates a transformed image. If the source image was at the root of the image path, it is saved as individual PDF file. Images in the same subdirectory are concatenated to one PDF file with the subdirectory name as file name. All created PDF files are stored in a newly created directory `pdfs` in the current working directory.

To install the script on a Linux system, make sure that all dependencies are fulfilled and run the following:

`chmod a+rx scanconv.py`
`sudo ln -s $(pwd)/scanconv.py /usr/local/bin/scanconv`
