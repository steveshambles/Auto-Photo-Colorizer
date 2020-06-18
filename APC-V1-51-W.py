"""
Auto Photo colourizer v1.51W
Steve Shambles. Updated June 18th 2020
stevepython.wordpress.com

Requirements:
pip3 install matplotlib
pip3 install numpy
pip3 install opencv-python
pip3 install pillow

files requird in root dir:
colorization_deploy_v2.prototxt
colorization_release_v2.caffemodel
pc-panel-280x105.png
pts_in_hull.npy

https://github.com/steveshambles/Auto-Photo-Colorizer
------------------------------------------------------
V1.51-Made sure only jpg,png and bmp files get processed.
      Lots of idiot checks implemented.
      Proper exit on clicking x on prg window.

V1-5- Inserted donate option and source code link.
      Linux version has probs now.

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

"""
import os
from pathlib import Path
import sys
from shutil import copyfile
from tkinter import Button, DISABLED, filedialog, Label, LabelFrame
from tkinter import messagebox, Menu, NORMAL, Tk, Toplevel
import webbrowser

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

root = Tk()
root.title('Auto Photo Colourizer V1.51W')
root.resizable(False, False)


def msg_box():
    """Custom non-blocking message box."""
    global custom_mbox
    custom_mbox = Toplevel(root)
    custom_mbox.title('APC V1.51W. ')
    custom_mbox.attributes('-topmost', 1)  # Bring custom window to front.

    # Remove the toolbar from the msg box.
    custom_mbox.attributes('-toolwindow', 1)
    # Iconify msg box if exit X is hit by user, rather than lose the window.
    custom_mbox.protocol('WM_DELETE_WINDOW', custom_mbox.iconify)

    cstm_mb_label = Label(custom_mbox, bg='indianred',
                          text='\n    Auto Photo Colourizer    \n\n'
                          '    Colourizing image....   \n'
                          '\nplease wait...\n')
    cstm_mb_label.grid()

    # Centre it.
    WINDOW_WIDTH = custom_mbox.winfo_reqwidth()
    WINDOW_HEIGHT = custom_mbox.winfo_reqheight()
    POSITION_RIGHT = int(custom_mbox.winfo_screenwidth()/2 - WINDOW_WIDTH/2)
    POSITION_DOWN = int(custom_mbox.winfo_screenheight()/2 - WINDOW_HEIGHT/2)
    custom_mbox.geometry('+{}+{}'.format(POSITION_RIGHT, POSITION_DOWN))

    root.update()


def load_img():
    """Load black and white image via file selector."""
    global users_image
    users_image = ''
    users_image = filedialog.askopenfilename(filetypes=(('All files', '*.*'),
                                                        ('Jpg', '*.Jpg'),
                                                        ('Png', '*.png'),
                                                        ('Bmp', '*.bmp')))
    if not users_image:
        return
    if users_image.endswith('.png') or users_image.endswith('.bmp')  \
        or users_image.endswith('.jpg'):

        # Display users original image.
        show_image = cv2.imread(users_image)
        # convert to gray in case sepia or something dodgy like that.
        show_image = cv2.cvtColor(show_image, cv2.COLOR_BGR2GRAY)
        show_image = cv2.resize(show_image, (640, 480))
        cv2.imshow('Original image temporarily resized to 640x480 with no aspect ratio.', show_image)
        cv2.waitKey(300)
        colorize_img_btn.configure(state=NORMAL)
    else:
        messagebox.showerror('File Error!:', 'Error, the file: \n\n'
                             + str(users_image) + '\n\n'
                             'is not a jpg, png or bmp.')


def save_img():
    """Save colourized image via file selector."""
    global colorized

    saved_image = 'noname'
    saved_image = filedialog.asksaveasfilename(title='Type a new name for your colourized image')

    if saved_image == '':
        saved_image = 'noname'

    saved_image = saved_image + str('.jpg')
    cv2.imwrite(saved_image, cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()
    load_img_btn.configure(state=NORMAL)
    colorize_img_btn.configure(state=DISABLED)
    save_img_btn.configure(state=DISABLED)


def missing_file_msg(file_name):
    """Missing files error message and download option."""
    root.withdraw()
    messagebox.showerror('Missing File:', 'Error, the file: \n\n'
                         + str(file_name) + '\n\n'
                         'is missing from the program directory.\n'
                         'Please replace file from my GitHub page.\n'
                         'and run APC again.'
                         )
    webbrowser.open('http://tinyurl.com/y9vymv5t')
    root.destroy()
    exit(0)


def check_files_exists():
    """Check model files are present."""
    if not os.path.isfile('colorization_deploy_v2.prototxt'):
        missing_file_msg('colorization_deploy_v2.prototxt')

    if not os.path.isfile('colorization_release_v2.caffemodel'):
        missing_file_msg('colorization_release_v2.caffemodel')

    if not os.path.isfile('pts_in_hull.npy'):
        missing_file_msg('pts_in_hull.npy')

    if not os.path.isfile('pc-panel-280x105.png'):
        missing_file_msg('pc-panel-280x105.png')


def colourize():
    """AI algo to colourize a black and white image."""
    global users_image
    global colorized

    # load_img_btn.configure(state=DISABLED)
    colorize_img_btn.configure(state=DISABLED)

    msg_box()

    # Define Model Paths:
    prototxt = 'colorization_deploy_v2.prototxt'
    model = 'colorization_release_v2.caffemodel'
    points = 'pts_in_hull.npy'

    # Load serialized black and white colorizer model and cluster.
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    # Add the cluster centers as 1x1 convolutions to the model.
    class8 = net.getLayerId('class8_ab')
    conv8 = net.getLayerId('conv8_313_rh')
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype('float32')]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]

    # Load the input image, scale it and convert it to Lab.
    image = cv2.imread(users_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Extracting “L”.
    scaled = image.astype('float32') / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Predicting “a” and “b”.
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # Creating a colorized Lab photo (L + a + b).
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Converting to RGB.
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype('uint8')
    plt.imshow(colorized)
    plt.axis('off')

    # Save.
    cv2.imwrite('temp-col-image.jpg',
                cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))

    # Display colourized image.
    img = cv2.imread('temp-col-image.jpg')
    img = cv2.resize(img, (640, 480))
    cv2.imshow('Colour image resized to 640x480. Original aspect ratio will be used when saved.', img)
    cv2.waitKey(300)
    save_img_btn.configure(state=NORMAL)

    custom_mbox.destroy()


def about_menu():
    """About program msgbox."""
    messagebox.showinfo('Program Information',
                        'Auto Photo Colourizer V1.51W\n\n'
                        'Freeware by Steve Shambles (c) June 2020\n\n'
                        'Please make a small donation\n'
                        'if you find this software useful.\nThankyou\n\n'
                        'Written in Python 3, using Opencv and Tkinter.\n\n')


def visit_blog():
    """Visit my python blog."""
    webbrowser.open('https://wp.me/pa5TU8-27P')


def contact_me():
    """Go to the contact page on my blog."""
    webbrowser.open('https://stevepython.wordpress.com/contact/')


def help_text():
    """Show help msg box."""
    messagebox.showinfo('How To Use Auto Photo Colourizer',
                        '1-First click on the LOAD IMAGE button.\n\n'
                        '2-From the file selector pick a jpg, png or bmp'
                        ' image from your drive.\n\n'
                        '3-Once your image is loaded and displayed,\n'
                        'click on the COLOURIZE button and wait.\n\n'
                        '4-After colourization has completed, and the colour'
                        ' image is displayed, \n'
                        '5-Click on the SAVE IMAGE button to save it.\n\n'
                        'Or you can close the image windows and load in your'
                        ' next image without saving.\n\n'
                        'Most colourized images will benefit from touching'
                        ' up in a program like Photoshop for best results.'
                        '\n\nIf you want to automatically colourize a folder '
                        'full of images, then select the first option in the '
                        'drop down menu "Auto colourize folder of images" '
                        'and choose a folder to process when presented with '
                        'the folder selector.'
                        '\n\nNon images in the folder will be ignored.\n\n'
                        'Only jpg, png and bmp images are recognized by APC.\n\n'
                        'A new photo will be colourized saved as a jpg file '
                        'and use the original filename but with the word '
                        '"colourized" tagged on the end.\n\n'
                        'Your original photos will not be changed or deleted.\n\n'
                        )


def donate_me():
    webbrowser.open("https:\\paypal.me/photocolourizer")


def visit_github():
    webbrowser.open("https://github.com/steveshambles/Auto-Photo-Colorizer")


def exit_apc():
    """Yes-no requestor to exit program."""
    ask_yn = messagebox.askyesno('Question',
                                 'Are you sure you want to exit APC?')
    if ask_yn is False:
        return
    root.destroy()
    sys.exit()


def color_folder_of_imgs():
    global users_image
    users_image = None

    # Open the tk folder requestor.
    folder_slctd = ''
    folder_slctd = filedialog.askdirectory(title='Please select a folder that contains only images')

    if not folder_slctd:
        return
    load_img_btn.configure(state=DISABLED)
    files = os.listdir(folder_slctd)

    for file in files:
        users_image = folder_slctd + '/' + file
        print(file)
        if file.endswith('.png') or file.endswith('.bmp')  \
           or file.endswith('.jpg'):

            colourize()

            fname = (os.path.basename(users_image))
            new_fname = (os.path.splitext(fname)[0])

            save_img_btn.configure(state=DISABLED)
            save_img = folder_slctd + '/' + new_fname + '-colorized.jpg'
            cv2.imwrite(save_img, cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
            cv2.destroyAllWindows()

    load_img_btn.configure(state=NORMAL)
    messagebox.showinfo('Information', 'Batch operation complete')
    webbrowser.open(folder_slctd)


# Make sure all files required are present.
check_files_exists()

# Insert logo.
logo_frame = LabelFrame(root)
logo_frame.grid(padx=5, pady=5)

logo_image = Image.open('pc-panel-280x105.png')
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = Label(logo_frame, image=logo_photo)
logo_label.logo_image = logo_photo
logo_label.grid(padx=2, pady=2)

# Buttons.
btns_frame = LabelFrame(root)
btns_frame.grid(padx=10, pady=10)

load_img_btn = Button(btns_frame, bg='green2', text='Load Img',
                      command=load_img)
load_img_btn.grid(pady=15, padx=15, column=0, row=0)

colorize_img_btn = Button(btns_frame, bg='skyblue', text='Colourize',
                          command=colourize)
colorize_img_btn.grid(pady=15, padx=15, column=1, row=0)

save_img_btn = Button(btns_frame, bg='salmon', text='Save Img',
                      command=save_img)
save_img_btn.grid(pady=15, padx=15, column=2, row=0)

# Drop-down menu.
menu_bar = Menu(root)
file_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label='Menu', menu=file_menu)
file_menu.add_command(label='Auto colourize folder of images', command=color_folder_of_imgs)
file_menu.add_command(label='Help', command=help_text)
file_menu.add_command(label='About', command=about_menu)
file_menu.add_separator()
file_menu.add_command(label='My Python Blog', command=visit_blog)
file_menu.add_command(label='APC Python source code', command=visit_github)
file_menu.add_command(label='Contact Me', command=contact_me)
file_menu.add_command(label='Make a small donation via PayPal', command=donate_me)
file_menu.add_separator()
file_menu.add_command(label='Exit', command=exit_apc)
root.config(menu=menu_bar)
root.eval('tk::PlaceWindow . Center')
root.protocol("WM_DELETE_WINDOW", exit_apc)

# Disable these buttons at startup.
colorize_img_btn.configure(state=DISABLED)
save_img_btn.configure(state=DISABLED)

root.mainloop()
