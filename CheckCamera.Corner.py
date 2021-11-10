import cv2
import matplotlib.pyplot as plt
import time
 
# カメラ準備 
cap = cv2.VideoCapture(0)
 
# 無限ループ 
while True:
    # キー押下で終了 
    key = cv2.waitKey(1)
    if key != -1:
        break
 
    # カメラ画像読み込み 
    ret, frame = cap.read()
 
    # 画像表示 
    cv2.imshow('image', frame)
 
# 終了処理 
cap.release()
cv2.destroyAllWindows()

# def main(device_num):
#     print("a")
#     cap = cv2.VideoCapture(device_num)
    
#     time.sleep(2)

#     ret, frame = cap.read()

#     plt.imshow(frame)
#     plt.show()


# if __name__ == "__main__":
#     device_num = 1
#     main(device_num)
