import time
from math import radians, cos, sin, asin, sqrt
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil

import torch
import cv2
import numpy as np

from geopy import distance

#membaca koneksi
import argparse

#from MAVProxy.modules.lib import mp_module
from pymavlink import mavutil
import sys, traceback

import orangdetection

class deteksi:
    def __init__(self):
        self.model = self.load()
        #self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)

        self.kamera()
        

    def load(self):
        #model = torch.hub.load('.', 'custom', 'yolov5s', source='local')
        model = torch.hub.load('.', 'custom', 'best.pt', source='local')
        #model = torch.load('best.pt', map_location='cpu')        
        return model

    def objek(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        # labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        # return labels, cord
        cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return cord

    # def class_to_label(self, x):
    #     """
    #     For a given label value, return corresponding string label.
    #     :param x: numeric label
    #     :return: corresponding string label
    #     """
    #     return self.classes[int(x)]

    def kotak_objek(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.circle(frame, (x1//2, y1//2), 4, (0, 255, 0), -1)
        self.lat.append(self.vehicle.location.global_relative_frame.lat)               #di bagian terdeteksi objek
        self.long.append(self.vehicle.location.global_relative_frame.long)
        self.jumlh_point+=1
        return frame

    def kamera(self):
        webcam = cv2.VideoCapture(0)

        while True:
            ret, image = webcam.read()

            (h, w) = image.shape[:2] #w:image-width and h:image-height
            cv2.circle(image, (w//2, h//2), 4, (0, 255, 0), -1)
            
            results = self.objek(image)
            image = self.kotak_objek(results, image)
            #(x, y) = imagse.shape[:2]
            #print(a, "\n", imagse)
            
            
            
            #print(self.model())

            #cv2.putText(image, "a", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

            
            cv2.namedWindow("window", cv2.WINDOW_FREERATIO)
            cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.imshow("window", image)

            alti = self.vehicle.location.global_relative_frame.alt

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if alti <= 15:
                self.mavlink()

    def mavlink(self):
        # parser = argparse.ArgumentParser(description='Commands vehicle using vehicle.simple_goto.')
        # parser.add_argument('--connect')
        # args = parser.parse_args()

        #connection_string = args.connect
        connection_string = connect('/dev/ttyUSB0', wait_ready=True, baud=57600)
        sitl = None

        #koneksi ke wahana
        print("Connecting to vehicle on: %s" % connection_string)
        self.vehicle = connect(connection_string, wait_ready=True)

        self.lat = []
        self.long = []
        #jumlh_point = 0
        self.jumlh_point = len(self.lat)
        self.i = 0
        self.alt = 20
        self.dist = 0

        self.dive()

    def dropping(self):
        msg = self.vehicle.message_factory.command_long_encode(
        0, 0,    # target_system, target_component
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO, #command
        0, #confirmation
        1,    # servo number
        1500,          # servo position between 1000 and 2000
        0, 0, 0, 0, 0)    # param 3 ~ 7 not used
        # send command to vehicle
        self.vehicle.send_mavlink(msg)

    def jarak_calc(self, lat1, lon1, lat2, lon2):
        #konversi ke radian
        # lon1 = radians(lon1)
        # lon2 = radians(lon2)
        # lat1 = radians(lat1)
        # lat2 = radians(lat2)

        # #harvesine
        # dlon = lon2 - lon1
        # dlat = lat2 - lat1
        # a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

        # c = 2 * asin(sqrt(a))

        # # radius bumi kilometer. 3956 mil
        # r = 6371

        # hasil = c * r
        # hasil = hasil * 1000
        point1 = (lon1, lat1)
        point2 = (lon2, lat2)
        hasil = distance.distance(point1, point2).meters

        return(int(hasil))

    # while True:         #while di deteksi objek kamera
    #     lat.append(vehicle.location.global_relative_frame.lat)               #di bagian terdeteksi objek
    #     long.append(vehicle.location.global_relative_frame.long)
    #     #jumlh_point+=1
    #     #if titik akhir grid: break
    #     break

    
    def dive(self):
        while True:
            point = LocationGlobalRelative(self.lat[i], self.long[i], self.alt)
            self.vehicle.simple_goto(point, groundspeed=15)
            lati = self.vehicle.location.global_relative_frame.lat
            longi = self.vehicle.location.global_relative_frame.long
            jarak = self.jarak_calc(lati, longi, self.lat[i], self.long[i])
            #jarak = get_distance_metres
            #beberapa meter dari point
            if jarak <= 3:
                point = LocationGlobalRelative(self.lat[i], self.long[i], 10)
                self.vehicle.simple_goto(point, groundspeed=15)
                while True:
                    alti = self.vehicle.location.global_relative_frame.alt
                    if alti == 10:
                        self.dropping()
                        i+=1
                        break
                    else:
                        continue
            elif i == self.jumlh_point:
                break
            else:
                continue

    #def kirim_data(self):
        

deteksi()

