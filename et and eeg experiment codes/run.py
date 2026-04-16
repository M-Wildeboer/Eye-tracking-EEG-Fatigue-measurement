# YETI24 slideshow experiment runner (pygame UI + state machine)
# Inputs:
#   - Stimuli/Stimuli.csv + stimulus images in Stimuli/
#   - Config.csv (camera indices, screen size, slide time, shuffle)
# Output:
#   - CSV log per participant with timestamps and gaze coordinates (screen + stimulus space)
#
# This script runs the full pipeline:
# Detect → Calibration → Validate → (per-stimulus) Quick offset → Stimulus recording

import logging as log
import os
import sys
import time

import pandas as pd
import pygame as pg
from pygame.locals import *
from eeg_api import EEGAPI

"""
Name of file containing the Stimuli. By switching the stimulus file,
different versions of the experiment can be tested. A shortened stimulus
file speeds up the testing cycles.
"""

import libyeti24 as yeti24
from libyeti24 import draw_text

"""
libyeti24 provides:
- YETI24: dual-camera eye tracking (feature extraction → regression → gaze estimate)
- Stimulus / StimulusSet: stimulus loading, scaling, preview rendering
- Calib: calibration target grid + drawing utilities
"""


def main():
    # Connect the eye tracker and check whether both cameras are available
    Yet = yeti24.YETI24(
    USB,
    SURF,)
    
    if not Yet.connected:
        log.error(f"YET could not connect with cameras {Yet.usb_L} and {Yet.usb_R}")
        sys.exit(1)
    else:
        size_L = Yet.frame_L.shape[1], Yet.frame_L.shape[0]
        size_R = Yet.frame_R.shape[1], Yet.frame_R.shape[0]

        log.info(
            f"YET connected. "
            f"Left camera={Yet.usb_L}, resolution={size_L}; "
            f"Right camera={Yet.usb_R}, resolution={size_R}; "
            f"write_fps={Yet.fps}"
        )
    
    # Create the EEG logger used during the experiment
    eeg = EEGAPI(port="COM3", baud=2_000_000, output_dir="Data_EEG", part_id=PART_ID)
        
    """
                      Connect to the two eye cameras using the USB indices from Config.csv (USB_L, USB_R).
    If connection fails, check available camera indices and update Config.csv accordingly.
    """
    
    # Load all stimuli listed in the stimulus csv file
    if os.path.isfile(STIM_PATH):
        STIMS = yeti24.StimulusSet(STIM_PATH)
        log.info(str(STIMS.n()) + " stimuli loaded")
        if SHUFFLE:
            STIMS.shuffle()
            log.info(" ... shaken, not stirred")
    else:
        log.error(STIM_PATH + " not found. CWD: " + os.getcwd())
        sys.exit()
    """
    Information about stimuli is collected from the Stimuli-csv file and
    used to create a StimulusSet object (which is a list of Stimuli objects).
    If this fails, your Stimuli.csv file is not in place (folder Stimuli),
    or you are running this program in interactive mode (like working with R).
    """

    ## Calibration screens
    # Create the full calibration grid and the one-point quick calibration
    Cal = yeti24.Calib(SURF)
    log.info("Calibration screen loaded with " + str(Cal.n()) + " targets")
    """
    A calibration object is created with a default 3x3 grid
    """

    QCal = yeti24.Calib(SURF, pro_positions=[0.5, 0.5])
    """
    The quick calibration is created with one center target. Target positions are given
    as proportions of the screen surface.
    """

    ## Initial State
    # Load the Haar cascade used to detect the eyes at the start of the experiment
    try:
        Yet.init_eye_detection(EYECASC)
        log.info("Eye detection initialized: " + str(Yet.connected))
    except:
        log.error("Eye detection could not be initialized with Haar cascade " + EYECASC)
        sys.exit()

    STATE = "Detect"
    trial_index = 0

    """
    Start in Detect: run Haar-cascade detection to locate eye ROIs for both cameras.
    The cascade file (haarcascade_eye.xml) must be present in the working directory.
    After successful detection, we proceed to Calibration using the detected ROIs.
    """

    # Define the output columns for click/performance data
    click_columns = [
        "Exp",
        "Part",
        "trial_index",
        "Stim",
        "stim_start_time",
        "click_time",
        "reaction_time",
        "click_x_stim",
        "click_y_stim",
    ]

    if not os.path.isfile(CLICK_FILE):
        pd.DataFrame(columns=click_columns).to_csv(CLICK_FILE, index=False)
        
    

    ## FAST LOOP
    # Main experiment loop: handle user input, state changes, processing, and screen updates
    while True:
        for event in pg.event.get():
            key_forward = event.type == KEYDOWN and event.key == K_SPACE
            key_back = event.type == KEYDOWN and event.key == K_BACKSPACE
            
            # Handle mouse clicks during stimulus presentation and store reaction/click data
            if STATE == "Stimulus" and event.type == MOUSEBUTTONDOWN and event.button == 1:
                Yet.stop_recording()
                eeg.log_event("stimulus_offset", PART_ID, trial_index, Stim.file)
                result = eeg.stop_measure(eye_df=Yet.data)
                click_x_screen, click_y_screen = event.pos

                click_x_stim = (click_x_screen - Stim.pos[0]) / Stim.scale
                click_y_stim = (click_y_screen - Stim.pos[1]) / Stim.scale

                if (
                    0 <= click_x_stim <= Stim.original_size[0]
                    and 0 <= click_y_stim <= Stim.original_size[1]
                ):
                    click_time = time.time()
                    reaction_time = click_time - t_stim_started

                    click_row = pd.DataFrame(
                        {
                            "Exp": [EXP_ID + EXPERIMENTER],
                            "Part": [PART_ID],
                            "trial_index": [trial_index],
                            "Stim": [Stim.file],
                            "stim_start_time": [t_stim_started],
                            "click_time": [click_time],
                            "reaction_time": [reaction_time],
                            "click_x_stim": [click_x_stim],
                            "click_y_stim": [click_y_stim],
                        }
                    )

                    click_row.to_csv(CLICK_FILE, mode="a", header=False, index=False)
                    Yet.data.to_csv(RESULT_FILE, index=False)

                    if STIMS.remaining() > 0:
                        Yet.reset_offsets()
                        STATE = "prepareStimulus"
                        log.info(STATE)
                    else:
                        STATE = "Thank You"
                        log.info(STATE)
            
            
            """
            The event handler loop is working off the queue of events
            that have arrived since the last visit. As it only uses two keys,
            we do a pre-classification into back and forward. This makes the following
            code easier to read. Also, if you wanted to change the back and forward keys,
            it can be changed in this one place.

            The following conditional creates state transitions (it really is only one).
            As we will see down below, frame processing and graphical display solely
            depend on the state.
            """

            # Interactive state transitions (IT) caused by user input
            if STATE == "Detect":
                if Yet.eye_detected and key_forward:
                    # lock in last detected ROI before leaving Detect
                    Yet.update_eye_frame()
                    STATE = "Calibration"

                """
                Detect → Calibration:
                Once both eyes are detected, Space advances to calibration.
                We keep the most recent detected ROIs as the starting point for feature extraction.
                """

            elif STATE == "Calibration":
                if key_forward:
                    Yet.update_frame()
                    Yet.update_eye_frame()
                    Yet.update_quad_bright()
                    Yet.record_calib_data(Cal.active_pos())
                    """
                    During the calibration the frame processing is shut off, which can be seen in the frame processing
                    section. On key press, one frame is captured, the quad-bright measures are taken and added
                    to the training set.

                    After that the program checks whether their are remaining calibration points.
                    If so, the calibration advances.
                    If the sequence is complete, the recorded calibration data is used for training the Yeti.
                    From this point on, Yet can produce eye coordinates relative to the screen.
                    """
                    if Cal.remaining() > 0:
                        Cal.next()
                        STATE = "Calibration"
                    else:
                        Yet.train()
                        STATE = "Validate"
                    log.info(STATE)
                elif key_back:
                    STATE = "Detect"
            elif STATE == "Validate":
                """
                After validation the program moves on prepareStimulus, which is an invisible state.
                An automatic transitional down below takes care of this step
                """
                if key_forward:
                    STATE = "prepareStimulus"
                elif key_back:
                    Cal.reset()
                    Yet.reset()
                    STATE = "Calibration"
            elif STATE == "Quick":
                """
                The Quick calibration state succeeds the invisble state prepareStimulus. One reason for that is
                that the quick calibration uses a blurred version of the stimulus (see Presentitionals),
                so it has to be available at this point.
                """
                if key_forward:
                    Yet.update_offsets(QCal.active_pos())
                    t_stim_started = time.time()
                    Yet.start_recording(PART_ID, trial_index)
                    eeg.start_measure(PART_ID, trial_index, Stim.file)
                    eeg.log_event("stimulus_onset", PART_ID, trial_index, Stim.file)
                    STATE = "Stimulus"

                    """
                    When the quick calibration is complete, Yet updates its offsets
                    and moves on to the stimulus presentation. When the user presses the key,
                    the time is started. That is why we want the stimulus already loaded at this point.
                    """
            elif STATE == "Thank You":
                if key_forward:
                    Yet.release()
                    pg.quit()
                    sys.exit()
            if event.type == QUIT:
                """
                This conditional makes sure the program quits gracefully at any time,
                when the user presses Escape or closes the window.
                """

                Yet.release()
                pg.quit()
                sys.exit()

        # Automatic transitions (AT) based on internal state or elapsed time
        """
        Automatic transitionals are used to react to internal states of the system.

        Here are two different examples of using ATs, completing a complex task and reacting to time.
        """

        if STATE == "prepareStimulus":
            ret, Stim = STIMS.next()
            if ret:
                trial_index += 1
                Stim.load(SURF)
                STATE = "Quick"
            else:
                log.error("Could not load next Stimulus")
                sys.exit()
                
                
            """
            This ATC completes a complex computation and immediatly moves on to the next state.

            Before a stimulus can be shown on the screen it has to be loaded from the hard drive
            and pre-processed. All this takes time which we don't want to confound with the presentation time.
            """

        if STATE == "Stimulus":
            elapsed_time = time.time() - t_stim_started

            # End the stimulus automatically when the maximum viewing time is reached
            if elapsed_time > SLIDE_TIME:
                Yet.data.to_csv(RESULT_FILE, index=False)
                eeg.log_event("stimulus_offset", PART_ID, trial_index, Stim.file)
                result = eeg.stop_measure(eye_df=Yet.data)
                Yet.stop_recording()

                if STIMS.remaining() > 0:
                    Yet.reset_offsets()
                    STATE = "prepareStimulus"
                    log.info(STATE)
                else:
                    STATE = "Thank You"
                    log.info(STATE)


        # FRAME PROCESSING

        """
        Frame processing basically is a data processing stack, also very much like
        a pipeline in R. However, in this case we first have to build the stack,
        and that is why the Yeti class exposes all individual processing steps.
        To make sure the processing steps are only computed once, all steps come
        as updatze functions. These set the respective attribute,
        which can later be retrieved multiple times without being calculated again.
        This saves processing time, but the developer has to know the steps
        and apply them correctly.
        """

        # Run only the processing steps that belong to the current state
        if STATE == "Detect":
            Yet.update_frame()
            Yet.detect_eye()
            if Yet.eye_detected:
                Yet.update_eye_frame()
            # print("new_frame:", Yet.new_frame, "frame is None:", Yet.frame is None)
            """
            During the eye detection state, the Yet frame is continiously updated
            and undergoes eye detection. This is a very computing intensive task.
            This is why we don't use it throughout the whole experiment.
            (And because it makes the signal more shakey).
            """
        elif STATE == "Validate" or STATE == "Quick":
            Yet.update_frame()
            Yet.update_eye_frame()
            Yet.update_quad_bright()
            Yet.update_eye_pos()
            """
            The Validate state runs through the whole prediction statck. A trained model is required.
            As a result, the draw() method becomes available, which is used in the respective Presentitional.
            """
        elif STATE == "Stimulus":
            """
            In the Stimulus state, the prediction stack is further extended to produce
            coordinates relative to the original stimulus dimensions (in pixel),
            as the stimulus may have bee= centered and scaled. Then the data is internally recorded by the
            Yet object.
            """
            Yet.update_frame()
            Yet.update_eye_frame()
            Yet.update_quad_bright()
            Yet.update_eye_pos()
            Yet.update_eye_stim(Stim)
            Yet.record(EXP_ID + EXPERIMENTER, PART_ID, trial_index, Stim.file)

        # PRESENTATION / DRAWING
        
        SURF.fill(BACKGR_COL)
        """
        Remember, we are in the fast while loop.
        With every round the display is refreshed by painting it over with the
        background color.
        """
        if STATE == "Detect":
            # 1) message based ONLY on detection flag
            if Yet.eye_detected:
                draw_text("Eye detected!", SURF, (0.1, 0.85), FONT)
                draw_text("Space to continue", SURF, (0.1, 0.9), Font)
            else:
                draw_text("Trying to detect an eye.", SURF, (0.1, 0.85), FONT)

            # 2) image: prefer eye_frame if available, else fallback to frame
            img_src = None
            if Yet.eye_detected and Yet.eye_frame is not None:
                img_src = Yet.eye_frame
            elif Yet.frame is not None:
                img_src = Yet.frame

            if img_src is not None:
                Img = yeti24.frame_to_surf(
                    img_src, (int(SURF_SIZE[0] * 0.5), int(SURF_SIZE[1] * 0.5))
                )
                SURF.blit(Img, (int(SURF_SIZE[0] * 0.25), int(SURF_SIZE[1] * 0.25)))

            """
            The Detect screen is dynamic in that it changes when an eye is detected,
            which can change from one Yet frame to the next.
            """
        elif STATE == "Calibration":
            Cal.draw()
            draw_text(
                "Look at the orange circle and press Space.", SURF, (0.1, 0.9), Font
            )
            """
            The Calib class brings its own draw function.
            """
        elif STATE == "Validate":
            draw_text("Space: continue", SURF, (0.1, 0.9), Font)
            draw_text("Backspace: redo the calibration.", SURF, (0.1, 0.95), Font)
            Yet.draw_follow(SURF)
            """
            The Yet class brings a draw function that produces a following dot.
            """
        elif STATE == "Stimulus":
            Stim.draw()
            """
            The Stimulus class brings its own draw function, which takes care
            of positioning and scaling of the original image.
            """
        elif STATE == "Quick":
            Stim.draw_preview()
            QCal.draw()
            #Yet.draw_follow(SURF)
            draw_text(
                "Look at the gray circle and press Space.", SURF, (0.05, 0.75), Font
            )
            """
            During the Quick state  a blurred preview of the stimulus is shown in the backgroud.
            This way, the quick calibration can also regard the difference in screen brightness.
            Under certain lighting conditions, a the change in screen brightness, when the stimulus is presented,
            can severely bias the measures, as it influences the brightness measures.
            """

        elif STATE == "Thank You":
            draw_text("Thank you for taking part!", SURF, (0.1, 0.5), FONT)
            draw_text(
                "Press Space to end the program. Data has been saved",
                SURF,
                (0.1, 0.8),
                Font,
            )

        # update the screen to display the changes you made
        pg.display.update()


def read_config(path="Config.csv"):
    # Read all experiment settings from Config.csv
    global USB
    global EXP_ID, EXPERIMENTER
    global SURF_SIZE, SLIDE_TIME, STIM_FILE, SHUFFLE
    global CONFIG

    CONFIG = dict()
    Tab = pd.read_csv(path)
    for index, row in Tab.iterrows():
        CONFIG[row[0]] = row[1]
    USB = (int(CONFIG["USB_L"]), int(CONFIG["USB_R"]))
    EXP_ID = str(CONFIG["EXP_ID"])
    EXPERIMENTER = str(CONFIG["EXPERIMENTER"])
    SURF_SIZE = (int(CONFIG["WIDTH"]), int(CONFIG["HEIGHT"]))
    SLIDE_TIME = float(CONFIG["SLIDE_TIME"])
    STIM_FILE = CONFIG["STIM_FILE"]
    SHUFFLE = bool(CONFIG["SHUFFLE"])


def setup():
    """
    Creates global variables for Yeti24 and changes the working directory
    """
    global EXP_ID, EXPERIMENTER
    global YETA, YETA_NAME
    global WD, STIM_DIR, STIM_PATH, RESULT_DIR, CLICK_DIR, PART_ID, RESULT_FILE, CLICK_FILE, EYECASC

    ## Meta data of Yeta
    YETA = 24
    YETA_NAME = "Yeti" + str(YETA)

    ## Paths and files
    # WD = os.path.dirname(sys.argv[0])
    WD = "."
    os.chdir(WD)
    """Working directory set to location of yeta_1.py"""

    STIM_DIR = os.path.join(WD, "Stimuli")
    """Directory where stimuli reside"""
    STIM_PATH = os.path.join(STIM_DIR, STIM_FILE)
    """CSV file describing stimuli"""
    RESULT_DIR = "Data"
    CLICK_DIR = "Data_clicks_RT"
    """Directory where results are written"""
    PART_ID = str(int(time.time()))
    """Unique participant identifier by using timestamps"""
    RESULT_FILE = os.path.join(
        RESULT_DIR, YETA_NAME + "_" + EXP_ID + EXPERIMENTER + PART_ID + ".csv"
    )

    CLICK_FILE = os.path.join(
        CLICK_DIR, YETA_NAME + "_" + EXP_ID + EXPERIMENTER + PART_ID + "_clicks.csv"
    )
    
    """File name for data"""
    EYECASC = "haarcascade_eye.xml"

    ##### Logging #####
    log.basicConfig(filename="Yet.log", level=log.INFO)


def init_pygame():
    # Initialize pygame, the main fullscreen display, and the fonts/colors
    pg.init()
    global FONT, Font, font
    global col_black, col_green, col_white, BACKGR_COL
    global SURF, SURF_SIZE

    SURF = pg.display.set_mode((0, 0), pg.FULLSCREEN)
    pg.display.set_caption(YETA_NAME)
    SURF_SIZE = SURF.get_size()

    FONT = pg.font.Font("freesansbold.ttf", int(min(SURF_SIZE) / 20))
    Font = pg.font.Font("freesansbold.ttf", int(min(SURF_SIZE) / 40))
    font = pg.font.Font("freesansbold.ttf", int(min(SURF_SIZE) / 60))


    # Colour definitions
    col_black = (0, 0, 0)
    col_green = (0, 255, 0)
    col_white = (255, 255, 255)

    BACKGR_COL = col_white

# Start-up sequence: load config, prepare paths/settings, initialize pygame, then run the experiment
read_config()
setup()
init_pygame()
main()
