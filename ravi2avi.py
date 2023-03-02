import numpy as np
import cv2

def nothing(x):
    pass

def ravi2avi(ravi_file_name, out_file_name):
    #ravi_file_name = 'K10 2021-03-25 9-17.ravi'
    
    cap = cv2.VideoCapture(ravi_file_name)  # Opens a video file for capturing

    # Fetch undecoded RAW video streams
    cap.set(cv2.CAP_PROP_FORMAT, -1)  # Format of the Mat objects. Set value -1 to fetch undecoded RAW video streams (as Mat 8UC1). [Using cap.set(cv2.CAP_PROP_CONVERT_RGB, 0) is not working]

    cols  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video frames width
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video frames height
    scale = 3


    ret, frame = cap.read()
    frame = frame.view(np.int16).reshape(rows, cols)
    frame_roi = frame[1:, :]
    width, height = frame_roi.shape

    l_v_f = np.min(frame_roi) #lower bound value for track slider
    h_v_f = np.max(frame_roi) #upper bound value for track slider
    l_v = 300
    h_v = 1000
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L-V", "Trackbars", l_v, 1000, nothing)
    cv2.createTrackbar("H-V", "Trackbars", h_v, 1000, nothing)

    print(l_v_f, h_v_f)

    l_s = 0
    h_s = 0

    
    while True:
        
        l_v = cv2.getTrackbarPos("L-V", "Trackbars")
        h_v = cv2.getTrackbarPos("H-V", "Trackbars")

        l_s = int((np.abs(h_v_f - l_v_f)*(l_v/1000))+l_v_f)
        h_s = int((np.abs(h_v_f - l_v_f)*(h_v/1000))+l_v_f)

        #print(l_s,h_s)
        #print((np.abs(h_v_f - l_v_f)))
        frame_roi2 = frame_roi.copy()
        frame_roi2[frame_roi2 <= l_s] = l_s
        frame_roi2[frame_roi2 >= h_s] = h_s

        normed = frame_roi2.copy()
        normed = (normed - l_s)
        normed = np.rint(((normed / np.abs(h_s - l_s))*.90)*255) #here we assign a 10% brightening buffer
        normed = normed.astype(np.uint8)
        #normed = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
        #print(normed, np.max(normed))

        normed = cv2.resize(normed, (cols*scale,rows*scale))
        cv2.imshow('normed', normed)  # Show the normalized video frame



        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    #cv2.destroyAllWindows()
    
    '''
    cap = cv2.VideoCapture(ravi_file_name)
    cap.set(cv2.CAP_PROP_FORMAT, -1)
    max_temp = -100000
    while True:
        ret, frame = cap.read() 
        if not ret:
            break
        
        frame = frame.view(np.int16).reshape(rows, cols)
        frame_roi = frame[1:, :]
        max_temp = np.max([max_temp, np.max(frame_roi)])

    
    cap.release()
    '''

    cap = cv2.VideoCapture(ravi_file_name)
    cap.set(cv2.CAP_PROP_FORMAT, -1) 


    max_temp = -1460
    max_temp_print = -1460

    #result = cv2.VideoWriter('c11_out.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, (width, height))
    #result = cv2.VideoWriter('c11_out.avi', cv2.VideoWriter_fourcc('M','J','P','G') ,10, (rows-1,cols),0)
    #result = cv2.VideoWriter('K10_out.mp4', cv2.VideoWriter_fourcc('M','J','P','G') ,100, (height, width),False)
    #result = cv2.VideoWriter('c11_out.mp4', cv2.VideoWriter_fourcc(*'mp4v') ,10, (height, width),False)

    # TO SAVE IN BLACK AND WHITE
    #result = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc('M','J','P','G') ,100, (height, width),False)

    # TO SAVE IN COLOUR
    #result = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'mp4v') ,10, (height, width),False)
    result = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc('M','J','P','G') ,10, (height, width))
    #cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, size)

    while True:
        #print('hello')
        ret, frame = cap.read()  # Read next video frame (undecoded frame is read as long row vector).

        if not ret:
            break  # Stop reading frames when ret = False (after the last frame is read).

        # View frame as int16 elements, and reshape to cols x rows (each pixel is signed 16 bits)
        frame = frame.view(np.int16).reshape(rows, cols)

        # It looks like the first line contains some data (not pixels).
        # data_line = frame[0, :]
        #frame_roi = frame[1:, :]  # Ignore the first row.
        frame_roi = frame[1:, :]
        max_temp_print = np.max([max_temp_print, np.max(frame_roi)])

        frame_roi2 = frame_roi.copy()
        frame_roi2[frame_roi2 <= l_s] = l_s
        frame_roi2[frame_roi2 >= h_s] = h_s

        normed = frame_roi2.copy()
        normed = (normed - l_s)
        #normed = np.rint(((normed / np.abs(h_s - l_s))*.45)*255) #here we assign a 10% brightening buffer
        normed = np.rint(((normed / np.abs(max_temp - l_s))*.95)*255) 
        #normed = np.rint(((normed / np.abs(h_v_f - l_s))*.95)*255) 
        normed = normed.astype(np.uint8)
        print(normed.shape, (width,height))
        

        # Normalizing frame to range [0, 255], and get the result as type uint8 (this part is used just for making the data visible)
        #print(np.min(frame_roi))
        #normed = cv2.normalize(frame_roi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #result.write(normed)
        #normed = cv2.resize(normed, (cols*scale,rows*scale))
        normed = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
        #normed_rgb = cv2.cvtColor(normed,cv2.COLORMAP_JET)
        result.write(normed)

        cv2.imshow('normed', cv2.resize(normed, (cols*scale,rows*scale)))  # Show the normalized video frame
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(max_temp_print)
    cap.release()
    result.release()
    cv2.destroyAllWindows()
