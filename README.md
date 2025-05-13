# Simplexity Demo

This demo showcases the Simplexity project.

To activate the demo for Simplexity, run “python main.py –show-webcam” on Terminal. 3 windows will pop-up – a text-based annotation UI showing the written content for a user to interact with, a webcam-based focus monitor showing real-time feedback on the perceived user state, and a dashboard indicating the overall concentration rate and a list of individual user states in real-time.
 
All dependent component scripts can be referenced within “main.py”.
 
“main_window.py” is the primary user interface for display of adaptive written content – it is a text window that has pop-ups when a user’s mouse cursor hovers over specific areas of the written content, with the pop-up differing based on captured user state at that point in time. To change the content displayed in “main_window.py”, go to “config.py” to edit both the original text displayed and annotative images and text.
 
“webcam_processor.py” contains the logic for the binary classification of webcam input into “Focused” and “Distracted” states, after computation of a distraction score which is currently based on a sum of blink rate, fixation average, and detection of eye contact. The settings are currently based on research findings on correlation of fixation average to distraction states, and custom tuning during development. You can further tune the binary classifier by changing the blink rate, fixation rate.
 
“dashboard_window.py” defines how the overall concentration rates of the users are displayed to a teacher. 
 

### What the Demo Does

Running `demo/main.py` will launch a PyQt6 application that consists of three main components displayed in separate windows:

1.  **Student Annotation Window**:
    *   Located on the right half of the screen.
    *   This is the primary interface for student input and interaction.
    *   It communicates with other components via OSC (Open Sound Control).

2.  **Webcam Viewer & Engagement Detection Window**:
    *   Located on the top-left quarter of the screen.
    *   Displays the live feed from the default webcam.
    *   Processes the webcam feed to detect student engagement levels using a pre-trained model (specified by the `--model-path` argument, defaulting to `best_model_v3.pth`).
    *   Sends engagement data to the Student Annotation Window via OSC.
    *   The webcam processing can be toggled with the `--show-webcam` command-line argument.

3.  **Dashboard Window**:
    *   Located on the bottom-left quarter of the screen.
    *   Listens for and displays data sent from the Student Annotation Window via OSC.
    *   This window serves as a monitoring tool for the data being generated or relayed by the student's application.

The application initializes OSC handlers for communication between these components and manages their layout on the screen.

### Command-line Arguments

*   `--show-webcam`: Use this flag to explicitly show the webcam processing window if it's not displayed by default or to ensure it's active.
*   `--model-path <path_to_model>`: Specifies the path to the engagement classifier model. Defaults to `best_model_v3.pth`. If the model is not found, a warning will be printed, and the system may use a fallback engagement detection method.

