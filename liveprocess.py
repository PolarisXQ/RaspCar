import glob
import io
import time
import threading
import picamera
import cv2
from PIL import Image
import numpy as np
import Car_control
import tensorflow as tf
from tensorflow import keras

resolution=(240,180)
car=Car_control.Car()

class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()
    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    # Read the image and do some processing on it
                    # Image.open(self.stream)
                    # print(self.stream)
                    image = Image.open(self.stream)
                    image_np = np.array(image)
                    canny = cv2.Canny(image_np, 10, 150)
                    canny=tf.expand_dims(canny,axis=0)
                    canny=tf.expand_dims(canny,axis=3)
                    prob=model.predict(canny)
                    action_num= np.argmax(prob)
                    #print(action_num)
                    control_car(action_num)
                    # Set done to True if you want the script to terminate
                    # at some point
                    #self.owner.done=True
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                      self.owner.pool.append(self)
class ProcessOutput(object):
    def __init__(self):
        self.done = False
        # Construct a pool of 4 image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self) for i in range(2)]
        self.processor = None
    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame; set the current processor going and grab
            # a spare one
            if self.processor:
                self.processor.event.set()
            with self.lock:
                if self.pool:
                    self.processor = self.pool.pop()
                else:
                    print("err1")
                    # No processor's available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)

    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        # Now, empty the pool, joining each thread as we go
        while True:
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    print("pool is empty")
                    #pass
            proc.terminated = True
            proc.join()

def control_car(action_num):
    if action_num == 0:
        print("Left")
        car.Car_Run(120,0)
        time.sleep(0.25)
    elif action_num == 1:
        print("Right")
        car.Car_Run(0,120)
        time.sleep(0.25)
    elif action_num == 2:
        print("Forward")
        car.Car_Run(120,120)
    elif action_num == 3:
        car.Car_Back(120, 120)
        print("Backward")
    else:
        car.Car_Stop()
        print('stop')

def main():
    ckpt_path='/home/pi/Desktop/checkpoint10'
    global model
    model=keras.models.load_model(ckpt_path)
    with picamera.PiCamera(resolution=resolution,framerate=20) as camera:
        camera.start_preview()
        time.sleep(2)
        output = ProcessOutput()
        camera.start_recording(output, format='mjpeg')
        while not output.done:
            camera.wait_recording(1)
        camera.stop_recording()

if __name__=='__main__':
    main()