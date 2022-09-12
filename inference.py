# !pip install retinaface

import os
from retinaface import RetinaFace
import cv2

# 2 Parameters
video_name = 'Sample.mp4'
expected_export_rate = 30

# code
vidCap = cv2.VideoCapture(video_name)
frame_count = vidCap.get(cv2.CAP_PROP_FRAME_COUNT)

frame_no = 0

os.mkdir(f'output/out_{video_name}')
id = 0
while vidCap.isOpened():
    ret, frame = vidCap.read()
    if not ret:
        break
    if frame_no % expected_export_rate == 0:
        resp = RetinaFace.detect_faces(frame)
        count = 0
        try:
            for face in resp.keys():
                count += 1
                area = resp[face]['facial_area']
                output = frame[area[1]:area[3], area[0]:area[2]]
                cv2.rectangle(frame, (area[0], area[1]),
                              (area[2], area[3]), (0, 0, 255), 2)
        except:
            pass
        cv2.imwrite(f'output/out_{video_name}/img_{id}.jpg', frame)
        id += 30
        print(id)
        print(
            f'{frame_no//expected_export_rate+1} frame exported! frame: {frame_no}/{int(frame_count)}')
    frame_no += 1
    if cv2.waitKey(1) == ord('x'):
        vidCap.release()
print('Done!')
