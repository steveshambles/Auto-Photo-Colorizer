# Auto-Photo-Colorizer
Python GUI to colorize black and white photos easily.
For Windows. Problems with Linux version at the moment.


![Alt text](https://stevepython.files.wordpress.com/2020/06/auto-photo-colourizer-v1-5w-logo.png "Optional title")


Auto Photo colourizer v1-5-W

By Steve Shambles. Updated June 17th 2020

stevepython.wordpress.com



Requirements:
--------------
pip3 install matplotlib

pip3 install numpy

pip3 install opencv-python

pip3 install pillow


files in root dir:
------------------
colorization_deploy_v2.prototxt

colorization_release_v2.caffemodel

pc-panel-280x105.png

pts_in_hull.npy

APC homepage:
-------------
https://stevepython.wordpress.com/2020/06/17/auto-photo-colorizer-v1-5w

V1-5-Inserted donate option and source code link


V1-4-Added option in menu to convert a folder of images automatically.


V1-3-replaced pop up box with custom non blocking popup during colorizing.
     centered main gui on screen.
     centred custom msg box on screen.
     convert img to gray frst in case sepia or something dodgy like that.


V1-2-Changed get missing files from dropbox to github.
     Message in title bars of images about temp aspect ratio.


V1.1-resized image displays to be 640x480 default, no aspect ratio yet,
     when saved it is original size though.
     Corrected bad spacing in help text pop up.
     pycodestyle linted.


