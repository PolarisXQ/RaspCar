import os
import cv2


def canny(path = "data\\",to_path= "cannied_data\\",threshold1=10,threshold2=120):
    files = os.listdir(path)                             # 将该路径下的文件名都存入列表
    #print(files)
    # turns = int(math.ceil(len(files) / CHUNK_SIZE))      # 取整，把所有图片分为这么多轮，每CHUNK_SIZE张一轮
    print("number of files: {}".format(len(files)))
    #print("turns: {}".format(turns))
    for step in range(len(files)):
        img=cv2.imread(path+files[step],0)
        # flag=-1时，8位深度，原通道
        # flag=0，8位深度，1通道
        # flag=1,   8位深度  ，3通道
        # flag=2，原深度，1通道
        # flag=3,  原深度，3通道
        # flag=4，8位深度 ，3通道
        cannied=cv2.Canny(img,threshold1,threshold2)
        # cv2.imshow("img",cannied)
        # cv2.waitKey(0)
        cv2.imwrite(to_path+files[step],cannied)

canny()