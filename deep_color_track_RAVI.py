import cv2
import numpy as np
import sys
from random import randint
import math
import pandas as pd
import argparse
import matplotlib
import io

def nothing(x):
    pass
def angle_trunc(a):
    while a < 0.0:
        a += math.pi * 2
    return a
    
def clean_deep_csv(deep_csv):
    with open(deep_csv, newline='') as csvfile:
    #spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #for row in spamreader:

        count = 0
        hearder_row1 = []
        header_out = []
        csv_mod = ""
        for row in csvfile:

            if count == 1:
                header_row1 = row.split(',')
            elif count ==2 :
                cols = row.split(',')
                for i in range(len(cols)):
                    header_out.append((header_row1[i]).rstrip()+'_'+cols[i])
                    print(header_out[i])

                csv_mod = csv_mod + (','.join(header_out))
            elif count > 2:
                csv_mod = csv_mod + row
                #print(row)

            count+=1
        return(csv_mod)
'''
parser = argparse.ArgumentParser()
parser.add_argument('video')
parser.add_argument('deeplabcut_csv')
parser.add_argument('frame_start')
parser.add_argument('file_out_name')

#checks to make sure a source file was given. If not, it prints the help display
if len(sys.argv)<=4:
    parser.print_help(sys.stderr)
    print("please provide needed arguments")
    print("video - file location of video")
    print("deeplabcut_csv - file location of deeplabcut csv output")
    print("frame_start - starting frame")
    print("file_out_name - the name of the output file")
    sys.exit(1)

args = parser.parse_args()
file_location = args.video
deep_csv = args.deeplabcut_csv
starting_frame = int(args.frame_start)
output_name = args.file_out_name
'''
def color_track(ravi_file_name, csv_file_name, out_file_name):

    cleaned_csv = clean_deep_csv(csv_file_name)
    df = pd.read_csv(io.StringIO(cleaned_csv))

    #####
    nose_x = df.nose_x.to_numpy()
    nose_y = df.nose_y.to_numpy()

    cement_x = df.cement_x.to_numpy()
    cement_y = df.cement_y.to_numpy()

    ear_left_x = df.ear_left_x.to_numpy()
    ear_left_y = df.ear_left_y.to_numpy()

    ear_right_x = df.ear_right_x.to_numpy()
    ear_right_y = df.ear_right_y.to_numpy()

    body_center_x = df.body_center_x.to_numpy()
    body_center_y = df.body_center_y.to_numpy()

    body_left_x = df.body_left_x.to_numpy()
    body_left_y = df.body_left_y.to_numpy()

    body_right_x = df.body_right_x.to_numpy()
    body_right_y = df.body_right_y.to_numpy()

    body_back_x = df.body_back_x.to_numpy()
    body_back_y = df.body_back_y.to_numpy()

    tail_x = df.tail_x.to_numpy()
    tail_y = df.tail_y.to_numpy()

    
    cap = cv2.VideoCapture(ravi_file_name)
    cap.set(cv2.CAP_PROP_FORMAT, -1) 
    cols  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video frames width
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video frames height


    hsv_data_frame_nose = []
    hsv_data_frame_cement = []
    hsv_data_frame_ear_left = []
    hsv_data_frame_ear_right = []
    hsv_data_frame_body_left = []
    hsv_data_frame_body_right = []
    hsv_data_frame_body_center = []
    hsv_data_frame_body_back = []
    hsv_data_frame_tail = []
    hsv_data_frame_max = []
    hsv_data_frame_min = []
    frame_count_arr = []
    frame_counter = 0

    

    while True:

        ret, frame_original = cap.read()
        if not ret or frame_counter >= len(nose_x):
            break
        
        frame_original = frame_original.view(np.int16).reshape(rows, cols)
        frame = frame_original[1:, :]

        normed = frame.copy()
        normed = (normed - np.nanmin(normed))
        normed = np.rint(((normed / np.abs(np.nanmax(normed) - np.nanmin(normed)))*.95)*255)
        normed = normed.astype(np.uint8)

        #frame = frame[0: 1000, 0: 2000]
        frame_nose = cv2.bitwise_and(frame, frame, mask=None) #copy of frame.
        frame_cement = cv2.bitwise_and(frame, frame, mask=None) #copy of frame.
        frame_ear_left = cv2.bitwise_and(frame, frame, mask=None) #copy of frame.
        frame_ear_right = cv2.bitwise_and(frame, frame, mask=None) #copy of frame.
        frame_body_left = cv2.bitwise_and(frame, frame, mask=None) #copy of frame.
        frame_body_right = cv2.bitwise_and(frame, frame, mask=None) #copy of frame.
        frame_body_center = cv2.bitwise_and(frame, frame, mask=None) #copy of frame.
        frame_body_back = cv2.bitwise_and(frame, frame, mask=None) #copy of frame.
        frame_tail = cv2.bitwise_and(frame, frame, mask=None) #copy of frame.
        frame_hot = cv2.bitwise_and(frame, frame, mask=None) #copy of frame.
        
        #gray_frame = cv2.cvtColor(normed, cv2.COLOR_BGR2GRAY)
        gray_frame = normed.copy()
        _, mask_n = cv2.threshold(gray_frame, 255, 255, cv2.THRESH_BINARY)
        _, mask_c = cv2.threshold(gray_frame, 255, 255, cv2.THRESH_BINARY)
        _, mask_el = cv2.threshold(gray_frame, 255, 255, cv2.THRESH_BINARY)
        _, mask_er = cv2.threshold(gray_frame, 255, 255, cv2.THRESH_BINARY)
        _, mask_bl = cv2.threshold(gray_frame, 255, 255, cv2.THRESH_BINARY)
        _, mask_br = cv2.threshold(gray_frame, 255, 255, cv2.THRESH_BINARY)
        _, mask_bc = cv2.threshold(gray_frame, 255, 255, cv2.THRESH_BINARY)
        _, mask_bb = cv2.threshold(gray_frame, 255, 255, cv2.THRESH_BINARY)
        _, mask_t = cv2.threshold(gray_frame, 255, 255, cv2.THRESH_BINARY)

        radius = 4
        
        ##region nose
        cv2.circle(mask_n, (int(nose_x[frame_counter]), int(nose_y[frame_counter])), int(radius), (255,255,255), -1)

        ##region cement
        cv2.circle(mask_c, (int(cement_x[frame_counter]), int(cement_y[frame_counter])), int(radius), (255,255,255), -1)

        ##region ear left
        cv2.circle(mask_el, (int(ear_left_x[frame_counter]), int(ear_left_y[frame_counter])), int(radius), (255,255,255), -1)

        ##region ear right
        cv2.circle(mask_er, (int(ear_right_x[frame_counter]), int(ear_right_y[frame_counter])), int(radius), (255,255,255), -1)

        ##region body left
        cv2.circle(mask_bl, (int(body_left_x[frame_counter]), int(body_left_y[frame_counter])), int(radius), (255,255,255), -1)

        ##region body right
        cv2.circle(mask_br, (int(body_right_x[frame_counter]), int(body_right_y[frame_counter])), int(radius), (255,255,255), -1)

        ##region body center
        cv2.circle(mask_bc, (int(body_center_x[frame_counter]), int(body_center_y[frame_counter])), int(radius), (255,255,255), -1)

        ##region body back
        cv2.circle(mask_bb, (int(body_back_x[frame_counter]), int(body_back_y[frame_counter])), int(radius), (255,255,255), -1)

        ##region tail
        cv2.circle(mask_t, (int(tail_x[frame_counter]), int(tail_y[frame_counter])), int(radius), (255,255,255), -1)
        
        #frames
        frame_nose = cv2.bitwise_and(frame_nose,frame_nose,mask = mask_n)
        frame_cement = cv2.bitwise_and(frame_cement,frame_cement,mask = mask_c)
        frame_ear_left = cv2.bitwise_and(frame_ear_left,frame_ear_left,mask = mask_el)
        frame_ear_right = cv2.bitwise_and(frame_ear_right,frame_ear_right,mask = mask_er)
        frame_body_left = cv2.bitwise_and(frame_body_left,frame_body_left,mask = mask_bl)
        frame_body_right = cv2.bitwise_and(frame_body_right,frame_body_right,mask = mask_br)
        frame_body_back = cv2.bitwise_and(frame_body_back,frame_body_back,mask = mask_bb)
        frame_body_center = cv2.bitwise_and(frame_body_center,frame_body_center,mask = mask_bc)
        frame_tail = cv2.bitwise_and(frame_tail,frame_tail,mask = mask_t)

        #print(np.nanmin(frame_nose))
        
        h_n = (frame_nose.sum(1).sum(0)/(frame_nose!=0).sum(1).sum(0))
        h_c = (frame_cement.sum(1).sum(0)/(frame_cement!=0).sum(1).sum(0))
        h_el = (frame_ear_left.sum(1).sum(0)/(frame_ear_left!=0).sum(1).sum(0))
        h_er = (frame_ear_right.sum(1).sum(0)/(frame_ear_right!=0).sum(1).sum(0))
        h_bl = (frame_body_left.sum(1).sum(0)/(frame_body_left!=0).sum(1).sum(0))
        h_br = (frame_body_right.sum(1).sum(0)/(frame_body_right!=0).sum(1).sum(0))
        h_bc = (frame_body_center.sum(1).sum(0)/(frame_body_center!=0).sum(1).sum(0))
        h_bb = (frame_body_back.sum(1).sum(0)/(frame_body_back!=0).sum(1).sum(0))
        h_t = (frame_tail.sum(1).sum(0)/(frame_tail!=0).sum(1).sum(0))
        h_max = np.nanmax(frame_hot)
        h_min = np.nanmin(frame_hot)
        
        '''
        hsv_frame_n = cv2.cvtColor(frame_nose, cv2.COLOR_BGR2HSV)
        hsv_frame_c = cv2.cvtColor(frame_cement, cv2.COLOR_BGR2HSV)
        hsv_frame_el = cv2.cvtColor(frame_ear_left, cv2.COLOR_BGR2HSV)
        hsv_frame_er = cv2.cvtColor(frame_ear_right, cv2.COLOR_BGR2HSV)
        hsv_frame_bl = cv2.cvtColor(frame_body_left, cv2.COLOR_BGR2HSV)
        hsv_frame_br = cv2.cvtColor(frame_body_right, cv2.COLOR_BGR2HSV)
        hsv_frame_bc = cv2.cvtColor(frame_body_center, cv2.COLOR_BGR2HSV)
        hsv_frame_bb = cv2.cvtColor(frame_body_back, cv2.COLOR_BGR2HSV)
        hsv_frame_t = cv2.cvtColor(frame_tail, cv2.COLOR_BGR2HSV)
        hsv_frame_hot = cv2.cvtColor(frame_hot, cv2.COLOR_BGR2HSV)

        h_n, s_n, v_n = hsv_frame_n[:, :, 0], hsv_frame_n[:, :, 1], hsv_frame_n[:, :, 2]
        h_c, s_c, v_c = hsv_frame_c[:, :, 0], hsv_frame_c[:, :, 1], hsv_frame_c[:, :, 2]
        h_el, s_el, v_el = hsv_frame_el[:, :, 0], hsv_frame_el[:, :, 1], hsv_frame_el[:, :, 2]
        h_er, s_er, v_er = hsv_frame_er[:, :, 0], hsv_frame_er[:, :, 1], hsv_frame_er[:, :, 2]
        h_bl, s_bl, v_bl = hsv_frame_bl[:, :, 0], hsv_frame_bl[:, :, 1], hsv_frame_bl[:, :, 2]
        h_br, s_br, v_br = hsv_frame_br[:, :, 0], hsv_frame_br[:, :, 1], hsv_frame_br[:, :, 2]
        h_bc, s_bc, v_bc = hsv_frame_bc[:, :, 0], hsv_frame_bc[:, :, 1], hsv_frame_bc[:, :, 2]
        h_bb, s_bb, v_bb = hsv_frame_bb[:, :, 0], hsv_frame_bb[:, :, 1], hsv_frame_bb[:, :, 2]
        h_t, s_t, v_t = hsv_frame_t[:, :, 0], hsv_frame_t[:, :, 1], hsv_frame_t[:, :, 2]
        h_hot, s_hot, v_hot = hsv_frame_t[:, :, 0], hsv_frame_t[:, :, 1], hsv_frame_t[:, :, 2]

        h_n = np.true_divide(h_n.sum(1),(h_n!=0).sum(1))
        s_n = np.true_divide(s_n.sum(1),(s_n!=0).sum(1))
        v_n = np.true_divide(v_n.sum(1),(v_n!=0).sum(1))

        h_c = np.true_divide(h_c.sum(1),(h_c!=0).sum(1))
        s_c = np.true_divide(s_c.sum(1),(s_c!=0).sum(1))
        v_c = np.true_divide(v_c.sum(1),(v_c!=0).sum(1))

        h_el = np.true_divide(h_el.sum(1),(h_el!=0).sum(1))
        s_el = np.true_divide(s_el.sum(1),(s_el!=0).sum(1))
        v_el = np.true_divide(v_el.sum(1),(v_el!=0).sum(1))

        h_er = np.true_divide(h_er.sum(1),(h_er!=0).sum(1))
        s_er = np.true_divide(s_er.sum(1),(s_er!=0).sum(1))
        v_er = np.true_divide(v_er.sum(1),(v_er!=0).sum(1))

        h_bl = np.true_divide(h_bl.sum(1),(h_bl!=0).sum(1))
        s_bl = np.true_divide(s_bl.sum(1),(s_bl!=0).sum(1))
        v_bl = np.true_divide(v_bl.sum(1),(v_bl!=0).sum(1))

        h_br = np.true_divide(h_br.sum(1),(h_br!=0).sum(1))
        s_br = np.true_divide(s_br.sum(1),(s_br!=0).sum(1))
        v_br = np.true_divide(v_br.sum(1),(v_br!=0).sum(1))

        h_bc = np.true_divide(h_bc.sum(1),(h_bc!=0).sum(1))
        s_bc = np.true_divide(s_bc.sum(1),(s_bc!=0).sum(1))
        v_bc = np.true_divide(v_bc.sum(1),(v_bc!=0).sum(1))

        h_bb = np.true_divide(h_bb.sum(1),(h_bb!=0).sum(1))
        s_bb = np.true_divide(s_bb.sum(1),(s_bb!=0).sum(1))
        v_bb = np.true_divide(v_bb.sum(1),(v_bb!=0).sum(1))

        h_t = np.true_divide(h_t.sum(1),(h_t!=0).sum(1))
        s_t = np.true_divide(s_t.sum(1),(s_t!=0).sum(1))
        v_t = np.true_divide(v_t.sum(1),(v_t!=0).sum(1))

        h_hot = np.true_divide(h_hot.sum(1),(h_hot!=0).sum(1))
        s_hot = np.true_divide(s_hot.sum(1),(s_hot!=0).sum(1))
        v_hot = np.true_divide(v_hot.sum(1),(v_hot!=0).sum(1))

        #h_p1 = 0
        #s_p1 = 0
        #v_p1 = 0


        cv2.imshow('frame n', frame_nose)
        cv2.imshow('frame hsv n', hsv_frame_n)
        cv2.imshow('frame',frame)




        hsv_data_frame_nose.append([np.nanmean(h_n), np.nanmean(s_n), np.nanmean(v_n)])
        hsv_data_frame_cement.append([np.nanmean(h_c), np.nanmean(s_c), np.nanmean(v_c)])
        hsv_data_frame_ear_left.append([np.nanmean(h_el), np.nanmean(s_el), np.nanmean(v_el)])
        hsv_data_frame_ear_right.append([np.nanmean(h_er), np.nanmean(s_er), np.nanmean(v_er)])
        hsv_data_frame_body_left.append([np.nanmean(h_bl), np.nanmean(s_bl), np.nanmean(v_bl)])
        hsv_data_frame_body_right.append([np.nanmean(h_br), np.nanmean(s_br), np.nanmean(v_br)])
        hsv_data_frame_body_center.append([np.nanmean(h_bc), np.nanmean(s_bc), np.nanmean(v_bc)])
        hsv_data_frame_body_back.append([np.nanmean(h_bb), np.nanmean(s_bb), np.nanmean(v_bb)])
        hsv_data_frame_tail.append([np.nanmean(h_t), np.nanmean(s_t), np.nanmean(v_t)])
        hsv_data_frame_hot.append([np.nanmin(h_hot), np.nanmin(s_hot), np.nanmin(v_hot)])
        frame_count_arr.append(frame_counter)
        '''

        cv2.imshow('frame n', normed)
        
        hsv_data_frame_nose.append([h_n])
        hsv_data_frame_cement.append([h_c])
        hsv_data_frame_ear_left.append([h_el])
        hsv_data_frame_ear_right.append([h_er])
        hsv_data_frame_body_left.append([h_bl])
        hsv_data_frame_body_right.append([h_br])
        hsv_data_frame_body_center.append([h_bc])
        hsv_data_frame_body_back.append([h_bb])
        hsv_data_frame_tail.append([h_t])
        hsv_data_frame_max.append([h_max])
        hsv_data_frame_min.append([h_min])
        frame_count_arr.append(frame_counter)

        frame_counter +=1


        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
        
        





    d_n = np.array( hsv_data_frame_nose).T *0.0373314 + 90.7215
    d_c = np.array( hsv_data_frame_cement).T *0.0373314 + 90.7215
    d_el = np.array( hsv_data_frame_ear_left).T *0.0373314 + 90.7215
    d_er = np.array( hsv_data_frame_ear_right).T *0.0373314 + 90.7215
    d_bl = np.array( hsv_data_frame_body_left).T *0.0373314 + 90.7215
    d_br = np.array( hsv_data_frame_body_right).T *0.0373314 + 90.7215
    d_bb = np.array( hsv_data_frame_body_back).T *0.0373314 + 90.7215
    d_bc = np.array( hsv_data_frame_body_center).T *0.0373314 + 90.7215
    d_t = np.array( hsv_data_frame_tail).T *0.0373314 + 90.7215
    d_max = np.array( hsv_data_frame_max).T *0.0373314 + 90.7215
    d_min = np.array( hsv_data_frame_min).T *0.0373314 + 90.7215

    fc = np.array( frame_count_arr)



    df2 = pd.DataFrame(fc)
    df2['nose'] = d_n[0]

    df2['cement'] = d_c[0]

    df2['ear_left'] = d_el[0]

    df2['ear_right'] = d_er[0]

    df2['body_left'] = d_bl[0]

    df2['body_right'] = d_br[0]

    df2['body_center'] = d_bc[0]

    df2['body_back'] = d_bb[0]

    df2['tail'] = d_t[0]

    df2['maxtemp'] = d_max[0]

    df2['mintemp'] = d_min[0]


    print(df2)

    destination_location = out_file_name + "_deep_color_track.csv"
    df2.to_csv(destination_location)
    
    
    # The following frees up resources and closes all windows
    cap.release()
    cv2.destroyAllWindows()
    