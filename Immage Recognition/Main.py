import cv2
import dxcam
import numpy as np
import time

# Set the template images & threshold
immages = ['Capture.png']
confidence_threshold = 0.7
cameraOutputColor = "GRAY"
cv2ConversionColor = cv2.IMREAD_GRAYSCALE
targetFPS = 60

#convert the templates immages to gray scale
templates = [cv2.imread(template, cv2ConversionColor) for template in immages]

# Create the DXCAM camera object
camera = dxcam.create(output_idx=0,
                      output_color=cameraOutputColor)
camera.start(target_fps=targetFPS)

# Initialize the FPS counter
start_time = time.time(); fps = 0


def Crop(leftCut, rightCut, Screenshot):
    top = 1080 - 120
    bottom = 1080
    width = Screenshot.shape[1]
    left = (width - leftCut) // 2
    right = left + rightCut
    return screenshot[top:bottom, left:right, :]

while True:
    # Get the latest frame from the camera
    screenshot = camera.get_latest_frame()

    #Use to crop if needed
    #screenshot = Crop(leftCut=240,  rightCut=240, Screenshot=screenshot)

    # Initialize a list to store matches for all templates
    all_matches = []

    # Perform template matching for each template
    for template in templates:
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)

        # Find locations above the confidence threshold
        locations = np.where(result >= confidence_threshold)
        matches = list(zip(*locations[::-1]))
        all_matches.extend(matches)

        # Draw bounding boxes around the matches
        for match in matches:
            top_left = match
            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
            cv2.rectangle(screenshot, top_left, bottom_right, (0, 255, 0), 1)
            
    #add FPS counter on the top left
    cv2.putText(screenshot, str(int(fps)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Screenshot", screenshot)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

    # Calculate the FPS
    current_time = time.time()
    fps = 1 / (current_time - start_time)
    start_time = current_time

# Stop the camera and close all windows
camera.stop()