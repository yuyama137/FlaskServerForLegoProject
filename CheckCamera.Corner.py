import cv2
import matplotlib.pyplot as plt
import time
 
# # カメラ準備 
# cap = cv2.VideoCapture(0)
 
# # 無限ループ 
# while True:
#     # キー押下で終了 
#     key = cv2.waitKey(1)
#     if key != -1:
#         break
 
#     # カメラ画像読み込み 
#     ret, frame = cap.read()
 
#     # 画像表示 
#     cv2.imshow('image', frame)
 
# # 終了処理 
# cap.release()
# cv2.destroyAllWindows()

def main(device_num):
    print("a")
    cap = cv2.VideoCapture(device_num)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    time.sleep(2)

    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print(frame.shape)

    plt.imshow(frame)
    plt.show()
    time.sleep(10)


if __name__ == "__main__":
    device_num = 1
    main(device_num)
