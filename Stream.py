import numpy as np
import cv2
from PIL import Image
import os
import datetime
import pathlib

def capture(frame_, dir_, name_):
    frame_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_, 'RGB')
    now = datetime.datetime.now()
    name = name_ + str(now.year) + str(now.month) + \
           str(now.day) + str(now.hour) + str(now.minute) + str(now.second) + str(now.microsecond)
    img.save(os.path.join(dir_, name + '.jpg'))

output_dir = "out"

cameras = ['ada',
           'gagarina',
           'pancevac',
           'autokomanda',
           'brankov',
           'gazela',
           'pupina',
           'despotastefana',
           'mostar',
           'trgrepublike',
           'slavija',
           'trgnikole',
           'takovska',
           'bogoslovija',
           'opstinanbg',
           'genex',
           'vuk']

def createCamerasFolder(parentFolder, cameraList):
    for camera in cameras:
        pathlib.Path(os.path.join(parentFolder, camera)).mkdir(parents=True, exist_ok=True)

link = 'https://streamer.devnet.rs/hls/'
extension = '.m3u8'

# vcap = cv2.VideoCapture('https://streamer.devnet.rs/hls/gazela.m3u8')
# if not vcap.isOpened():
#    print "File Cannot be Opened"

i = 0
flag = False

createCamerasFolder(output_dir, cameras)

while not flag:
    camera = cameras[int(np.random.random(1)*len(cameras))]
    vcap = cv2.VideoCapture(link + camera + extension)
    while vcap.isOpened():
        ret, frame = vcap.read()
        # print cap.isOpened(), ret
        if ret and frame is not None:
            cv2.imshow('frame', frame)
            i += 1
            if i == 50:
                capture(frame, os.path.join(output_dir, camera), camera)
                i = 0
            # Press q to close the video windows before it ends if you want
            if cv2.waitKey(22) & 0xFF == ord('q'):
                flag = True
                break
        else:
            print("Frame is None")
            break

    # When everything done, release the capture
    vcap.release()

cv2.destroyAllWindows()
print("Video stop")
