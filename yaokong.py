import Car_control
import pygame
import time
import threading
import io
import picamera

duration = 5
# seconds
resolution = (240, 180)


class SplitFrames(object):

    def __init__(self):
        self.frame_num = 0
        self.output = None

    # 处理图像的函数write
    # 对视频拍摄的每一帧进行处理，构造一个自定义输出类，每拍摄一帧都会进来write处理
    def write(self, buf):
        global key
        if buf.startswith(b'\xff\xd8'):  # 代表一个JPG图片的开始，新照片的开头
            # Start of new frame; close the old one (if any) and
            # open a new output
            if self.output:
                self.output.close()
            self.frame_num += 1
            self.output = io.open(
                'data/%s_%s_%s.jpeg' % (key, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), time.time()),
                'wb')  # 改变格式为jpg
        self.output.write(buf)


def main():
    car = Car_control.Car()
    pygame.init()
    pygame.display.set_mode((100, 100))  # 窗口
    car.Car_Stop()
    time.sleep(0.1)

    print("Start control!")
    global key
    key = 0
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key_input = pygame.key.get_pressed()
                print(key_input[pygame.K_w], key_input[pygame.K_a], key_input[pygame.K_d])
                # 按下前进，保存图片以2开头
                if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Forward")
                    key = 2
                    car.Car_Run(180, 180)
                    time.sleep(0.1)
                # 按下左键，保存图片以0开头
                elif key_input[pygame.K_a]:
                    print("Left")
                    car.Car_Run(0, 150)
                    time.sleep(0.1)
                    key = 0
                # 按下d右键，保存图片以1开头
                elif key_input[pygame.K_d]:
                    print("Right")
                    car.Car_Run(150, 0)
                    time.sleep(0.1)
                    key = 1
                # 按下s后退键，保存图片为3开头
                elif key_input[pygame.K_s]:
                    print("Backward")
                    car.Car_Back(150, 150)
                    key = 3
                # 按下k停止键，停止
                elif key_input[pygame.K_p]:
                    car.Car_Stop()
                elif key_input[pygame.K_u]:  # up
                    car.Ctrl_Servo(2, 0)
                elif key_input[pygame.K_b]:  # bow
                    car.Ctrl_Servo(2, 90)
                # elif key_input[pygame.K_o]:
                #     shot_thread = threading.Thread(target=car.shot)
                #     shot_thread.start()
                elif key_input[pygame.K_n]:
                    capture_thread = threading.Thread(target=capture)
                    capture_thread.start()
            elif event.type == pygame.KEYUP:
                car.Car_Stop()


def capture():
    with picamera.PiCamera(resolution=resolution, framerate=30) as camera:
        time.sleep(2)
        print("Start capture!")
        output = SplitFrames()
        start = time.time()
        camera.start_recording(output, format='mjpeg')
        camera.wait_recording(duration)
        camera.stop_recording()
        finish = time.time()
    print('Captured %d frames at %.2ffps' % (
        output.frame_num,
        output.frame_num / (finish - start)))
    print("end capture.")
    return


if __name__ == '__main__':
    main()