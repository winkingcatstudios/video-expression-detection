import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import emotion_logger

# create the root window
root = tk.Tk()
root.title('Video Emotion Logger')
root.resizable(False, False)
root.geometry('300x150')

# create emotion logger instance


def select_file():
    filetypes = (
        ('Compatible files', '*.mp4 *.avi *.mov *.wmv'), # Zoom defaults to .mp4 but other options are allowable for this program
    )

    load_label.pack(expand=True)

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    
    emotion_logger.set_vid_path(filename)
    load_label.config(text = "Loading Complete")
    run_button.pack(expand=True)

def logger_run():
    emotion_logger.run()
    root.geometry('300x500')
    result_label.config(text='-Results-'
    '\nAngry: ' + str(emotion_logger.angry) + 
    '\nDisgust: ' + str(emotion_logger.disgust) + 
    '\nFear: ' + str(emotion_logger.fear) + 
    '\nHappy: ' + str(emotion_logger.happy) + 
    '\nSad: ' + str(emotion_logger.sad) + 
    '\nSurprise: ' + str(emotion_logger.surprise) + 
    '\nNeutral: ' + str(emotion_logger.neutral) + 
    '\nDominant Emotion: ' + str(emotion_logger.dom_emo_string))
    result_label.pack(expand=True)

# instruction label
inst_label = tk.Label(
    root,
    text='Video Emotion Logger',
    width=30,
    font=('times', 18, 'bold')
)  

# open button
open_button = ttk.Button(
    root,
    text='Open a Video File',
    command=select_file
)

# loading label
load_label = tk.Label(
    root,
    text='Loading...',
    width=30,
    font=('times', 12)
)  

# run button
run_button = ttk.Button(
    root,
    text='Run Emotion Logger',
    command=logger_run,
)

# result label
result_label = tk.Label(
    root,
    text='-Results-',
    width=30,
    font=('times', 18, 'bold')
)  

inst_label.pack(expand=True)
open_button.pack(expand=True)



# run the application
root.mainloop()
