# import cv2 as cv
# import numpy as np
# import time
# import matplotlib.pyplot as plt

# fig, ax1 = plt.subplots(facecolor=(.18, .31, .31))
# ax1 = plt.subplot(111)
# ax1.set_facecolor('#eafff5')

# # The video feed is read in as a VideoCapture object
# #cap = cv.VideoCapture("shibuya.mp4")
# cap = cv.VideoCapture("satara_flw3.mp4")
# #cap = cv.VideoCapture("Car.mp4")
# # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
# ret, img= cap.read()
# height , width , layers =  img.shape
# xL = 640
# yL = int( height * xL/width)
# print("width, height, apr:", width, height, width/height)
# print("xL, yL, aspr:", xL, yL, xL/yL)

# ptx = []
# pty = []


# while True:
#     ret, vv= cap.read()
#     vv = cv.resize(vv, (xL, yL)) 
#     cv.namedWindow("vv", flags= cv.WINDOW_AUTOSIZE) 
#     cv.imshow("vv", vv)
#     if cv.waitKey(20) & 0xFF == ord('q'):
#         break

# cv.destroyWindow("vv")
# ret, first_frame = cap.read()
# first_frame = cv.resize(first_frame, (xL, yL))

# # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
# prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# # Creates an image filled with zero intensities with the same dimensions as the frame
# mask = np.zeros_like(first_frame)
# # Sets image saturation to maximum
# mask[..., 1] = 255

# def Get_x1_x2(event, x, y, flags, param):
#     #global ptx
#     #global pty
#     print(x,y)
#     time.sleep(.5)
#     if event == cv.EVENT_LBUTTONDOWN:
# 		#refPt = [(x, y)]
# 		#cropping = True
#         print("LBUTTON DOWN")
#         print("x, y:", x,y)
#         if len(ptx) == 2:
#             ptx.clear()
#             pty.clear()
#         ptx.append(x)
#         pty.append(y)
#         print("pt:", ptx, pty)

# while(cap.isOpened()):
#     # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
#     ret, frame = cap.read()
#     frame = cv.resize(frame, (xL, yL))
#    # ret, frame = cap.read()
#    # ret, frame = cap.read()
#     # Converts each frame to grayscale - we previously only converted the first frame to grayscale
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # Calculates dense optical flow by Farneback method
#     # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
#     prev_gray= prev_gray * 40 #20 #10 #20 #10 ok
#     #flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None,
#                    #0.5, 3, 15, 3, 5, 1.2, 0)#ok
#     #fflw = 0.8
#     flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None,
#                    0.5, 3, 15, 3, 5, 1.2, cv.OPTFLOW_FARNEBACK_GAUSSIAN)
    
#    # flow = cv.calcOpticalFlowPyrLK(prev_gray*2, gray*2, None,
#                #(8,8), 3, 15, 3, 5, 1.2,0) #cv.OPTFLOW_FARNEBACK_GAUSSIAN)


#     # Computes the magnitude and angle of the 2D vectors
#     magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
#     # Sets image hue according to the optical flow direction
#     mask[..., 0] = angle * 180 / np.pi / 2
#     # Sets image value according to the optical flow magnitude (normalized)
#     mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
#     Var = mask[...,2]
#     Aar = mask[...,0]
#     """
#     print("Var_shape:", Var.shape)
#     print("Var:\n", Var[200, 300:320])
#     print("Aar:\n", Aar[200, 300:320])
#     """
#     # Converts HSV to RGB (BGR) color representation
#     rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
#     #?h,s,v = cv.split(mask)
#     #?g_rgb = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
#     # Opens a new window and displays the output frame
#     thickness = 2 #1
#     color = [0, 255,0]
#     color1 = [0,0,255]
#     #yy = 700
#     for yy in range (100,300, 20):
#         for x in range (150, 400, 10):
#             VarL = Var[yy, x]
#             VarL = int(VarL/2)  #?????
#             ag =  Aar[yy,x] * np.pi /180
#             u = int(abs(VarL) * np.cos(ag))
#             v = int(abs(VarL) * np.sin(ag))
#             #start_point = (x, 400)
#             #end_point = (x, 400+arL)
#             #?print("Varl:", Var[yy, x])
#             rgb = cv.arrowedLine(rgb, (x, yy), (x, yy+ VarL),  
#                       color,  thickness, tipLength = 0.1)
#             #?rgb = cv.arrowedLine(rgb, (x, yy), (x+u, yy+v),  
#                      #? color1,  thickness, tipLength = 0.1)
#     cv.imshow("dense optical flow", rgb) #h*v/30) #g_rgb) #rgb)
#     #?cv.imshow("input", frame)
#     # Updates previous frame
#     prev_gray = gray

#     ###? to select the x1 and x2 point along the channel
    
#     if len(ptx) == 2:
#         file_name = "vel_data.txt"
#         f= open(file_name,"w+")
#         #f.close()
#         x1 = ptx[0]
#         x2 = ptx[1]
#         y1 = pty[0]
#         y2 = pty[1]
#         #cv.line(clone, (x1,y1), (x2,y1), color =(255,255,0), thickness=1)
        
#         i = 0
      
#         for xx in range (x1, x2, 10):
#             i = i+1
#             VarL = Var[y1, xx]
#             VarL = int(VarL/2) #*2)  #?????
#             ag =  Aar[y1,xx] * np.pi /180
#             u = int(abs(VarL) * np.cos(ag))
#             v = int(abs(VarL) * np.sin(ag))
        
#             frame = cv.arrowedLine(frame, (xx, y1), (xx, y1+ VarL),  
#                       color,  thickness=2) #, tipLength = 0.1)
#             #frame = cv.arrowedLine(frame, (xx, y1), (xx+u, y1+v),  
#                      # color1,  thickness=2, tipLength = .1)
#             print('speed:', v)
#             str_para_rd = str(xx)+','+str(y1)+','+str(VarL)
#             #f= open(file_name,"a+")
#             f.write(str(str_para_rd+"\n"))
#         f.close()

#     clone = frame.copy() 
#     #clone = cv.resize(clone, (xL,yL))
#     cv.namedWindow("clone", flags= cv.WINDOW_AUTOSIZE)
#     cv.setMouseCallback("clone", Get_x1_x2)
#     if len(ptx) == 2:
#         cv.line(clone, (x1,y1), (x2,y1), color =(255,255,0), thickness=1)
#     cv.imshow("clone", clone)
    
#    # frame.bind('<Button-1>',Get_x1_x2)

#     ###?
    
#     # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
#     if cv.waitKey(1) & 0xFF == ord('q'):
#          break
# # The following frees up resources and closes all windows
# cap.release()
# cv.destroyAllWindows()

import cv2 as cv
import numpy as np

# ----------------------------
# Settings
# ----------------------------
VIDEO_PATH = "satara_flw3.mp4"   # change if needed
OUT_W = 640                     # output width
SPEED_SCALE = 0.5               # like VarL/2 in your code
LINE_STEP_X = 10                # spacing along selected line
GRID_Y_RANGE = (100, 300)       # for visualization grid
GRID_X_RANGE = (150, 400)
GRID_STEP_Y = 20
GRID_STEP_X = 10

# ----------------------------
# Globals for click selection
# ----------------------------
ptx, pty = [], []
selection_ready = False
saved_for_this_selection = False

def on_mouse(event, x, y, flags, param):
    """Left click twice on the 'clone' window to select A then B."""
    global ptx, pty, selection_ready, saved_for_this_selection

    if event == cv.EVENT_LBUTTONDOWN:
        print(f"Clicked: ({x}, {y})")

        # if already 2 points, start new selection
        if len(ptx) == 2:
            ptx.clear()
            pty.clear()
            selection_ready = False
            saved_for_this_selection = False

        ptx.append(int(x))
        pty.append(int(y))

        if len(ptx) == 2:
            selection_ready = True
            saved_for_this_selection = False
            print("Selection ready: A =", (ptx[0], pty[0]), "B =", (ptx[1], pty[1]))
            print("Sampling + saving vel_data.txt now (once per selection).")

# ----------------------------
# Open video
# ----------------------------
cap = cv.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

ret, first = cap.read()
if not ret or first is None:
    raise RuntimeError("Could not read first frame. Check video path/codec.")

h, w = first.shape[:2]
OUT_H = int(h * OUT_W / w)
print("Input w,h:", w, h, " | Output W,H:", OUT_W, OUT_H)

# Reset to start
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Initialize prev_gray
ret, first_frame = cap.read()
if not ret or first_frame is None:
    raise RuntimeError("Could not read first frame after reset.")
first_frame = cv.resize(first_frame, (OUT_W, OUT_H))
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# HSV mask for flow visualization
mask = np.zeros_like(first_frame)
mask[..., 1] = 255  # saturation max

# ----------------------------
# Create windows ONCE and attach callback
# ----------------------------
cv.namedWindow("clone", cv.WINDOW_NORMAL)  # CLICK IN THIS WINDOW
cv.namedWindow("dense optical flow", cv.WINDOW_NORMAL)
cv.setMouseCallback("clone", on_mouse)

print("\nINSTRUCTIONS:")
print("1) Make sure the window titled 'clone' is focused.")
print("2) Left-click point A, then left-click point B in the 'clone' window.")
print("3) Press 'q' to quit.\n")

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("End of video / read error.")
        break

    frame = cv.resize(frame, (OUT_W, OUT_H))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Dense optical flow (Farneback)
    flow = cv.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2,
        cv.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # HSV: hue=direction, value=magnitude
    mask[..., 0] = ang * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    Var = mask[..., 2]  # 0..255
    Aar = mask[..., 0]  # 0..180 approx

    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

    # Draw a grid of arrows for visualization
    y0, y1 = GRID_Y_RANGE
    x0, x1 = GRID_X_RANGE
    for yy in range(y0, min(y1, OUT_H), GRID_STEP_Y):
        for xx in range(x0, min(x1, OUT_W), GRID_STEP_X):
            VarL = int(Var[yy, xx] * SPEED_SCALE)
            rgb = cv.arrowedLine(rgb, (xx, yy), (xx, yy + VarL), (0, 255, 0), 1, tipLength=0.2)

    cv.imshow("dense optical flow", rgb)

    # Show clone window for clicking
    clone = frame.copy()

    # If selected 2 points, draw line + sample along it
    if selection_ready and len(ptx) == 2:
        xa, xb = sorted([ptx[0], ptx[1]])
        y_sel = pty[0]  # keep y from first click (matches your original idea)

        # clamp inside image
        xa = int(np.clip(xa, 0, OUT_W - 1))
        xb = int(np.clip(xb, 0, OUT_W - 1))
        y_sel = int(np.clip(y_sel, 0, OUT_H - 1))

        cv.line(clone, (xa, y_sel), (xb, y_sel), (255, 255, 0), 2)

        # Save ONCE per selection
        if not saved_for_this_selection:
            with open("vel_data.txt", "w") as f:
                for xx in range(xa, xb + 1, LINE_STEP_X):
                    VarL = int(Var[y_sel, xx] * SPEED_SCALE)
                    f.write(f"{xx},{y_sel},{VarL}\n")
            print("Saved vel_data.txt")
            saved_for_this_selection = True

        # Draw arrows along selected line
        for xx in range(xa, xb + 1, LINE_STEP_X):
            VarL = int(Var[y_sel, xx] * SPEED_SCALE)
            clone = cv.arrowedLine(clone, (xx, y_sel), (xx, y_sel + VarL), (0, 255, 0), 2, tipLength=0.2)

        # Overlay point markers A and B
        cv.circle(clone, (ptx[0], pty[0]), 6, (0, 0, 255), -1)
        cv.circle(clone, (ptx[1], pty[1]), 6, (0, 0, 255), -1)

    cv.imshow("clone", clone)

    prev_gray = gray

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
