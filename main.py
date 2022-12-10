import torch
import cv2
import numpy as np

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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

deteksi()