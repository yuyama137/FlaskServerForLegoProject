import cv2
import numpy as np
import matplotlib.pyplot as plt

#抽出する面積のしきい値
AREA_RATIO_THRESHOLD = 0.00005
font = cv2.FONT_HERSHEY_SIMPLEX

def calc_area(h,w,contours):
    #面積を計算
    areas = np.array(list(map(cv2.contourArea,contours)))
    if len(areas) == 0 or np.max(areas) / (h*w) < AREA_RATIO_THRESHOLD:
        #見つからなかったらNoneを返す
        return None
    else:
        #面積が最大の塊の重心を計算し返す
        max_idx = np.argmax(areas)
        max_area = areas[max_idx]
        result = cv2.moments(contours[max_idx])
        x = int(result["m10"]/result["m00"])
        y = int(result["m01"]/result["m00"])
        return (x,y)

def detect_value(frame):
    h,w,c = frame.shape#高さ,幅,チャンネル数
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#hsv色空間に変換
    LOW_BLACK = np.array([0, 0, 0])
    HIGH_BLACK = np.array([360, 100, 270])
    ex_img = cv2.inRange(hsv,LOW_BLACK,HIGH_BLACK)#色を抽出する
    #マスキング処理
    masked_img = cv2.bitwise_and(frame, frame, mask=ex_img)
    cv2.imwrite("value_mask_img.png", ex_img)
    cv2.imwrite("value_masked_img.png", masked_img)
    _, contours,hierarchy = cv2.findContours(ex_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#輪郭抽出
    ans = calc_area(h,w,contours)
    return ans

def detect_red_color(frame):
    h,w,c = frame.shape#高さ,幅,チャンネル数
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#HSV色空間に変換
    #赤色のHSVの値域1
    hsv_min = np.array([0,64,0])
    hsv_max = np.array([10,255,255])
    ex_img_1 = cv2.inRange(hsv, hsv_min, hsv_max)
    #赤色のHSVの値域2
    hsv_min = np.array([160,64,0])
    hsv_max = np.array([179,255,255])
    ex_img_2 = cv2.inRange(hsv, hsv_min, hsv_max)
    ex_img=ex_img_1+ex_img_2
    _, contours,hierarchy = cv2.findContours(ex_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#輪郭抽出
    ans = calc_area(h,w,contours)
    return ans

def detect_blue_color(frame):
    h,w,c = frame.shape#高さ,幅,チャンネル数
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#hsv色空間に変換
    LOW_BLUE = np.array([100, 75, 0])
    HIGH_BLUE = np.array([140, 255, 255])
    ex_img = cv2.inRange(hsv,LOW_BLUE,HIGH_BLUE)#色を抽出する
    #masked_img = cv2.bitwise_and(frame, frame, mask=ex_img)
    #cv2.imwrite("blue_masked_img.png", masked_img)
    _, contours,hierarchy = cv2.findContours(ex_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#輪郭抽出
    ans = calc_area(h,w,contours)
    return ans

def detect_yellow_color(frame):
    h,w,c = frame.shape#高さ,幅,チャンネル数
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#hsv色空間に変換
    LOW_YELLOW = np.array([30,100,100])
    HIGH_YELLOW = np.array([40,255,255])
    ex_img = cv2.inRange(hsv,LOW_YELLOW,HIGH_YELLOW)#色を抽出する
    _, contours,hierarchy = cv2.findContours(ex_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#輪郭抽出
    ans = calc_area(h,w,contours)
    return ans

def detect_green_color(frame):
    h,w,c = frame.shape#高さ,幅,チャンネル数
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#hsv色空間に変換
    LOW_GREEN = np.array([40,64,0])
    HIGH_GREEN = np.array([80,255,255])
    ex_img = cv2.inRange(hsv,LOW_GREEN,HIGH_GREEN)#色を抽出する
    _, contours,hierarchy = cv2.findContours(ex_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#輪郭抽出
    ans = calc_area(h,w,contours)
    return ans

def _calc_dist(pt1, pt2):
    return np.sqrt(abs(pt1[0] - pt2[0])**2 + abs(pt1[1] - pt2[1])**2)

def _get_transform_util(top_left,top_right,bottom_left,bottom_right):
    """透視変換用の値と行列を取得"""
    left = _calc_dist(top_left, bottom_left)
    btm = _calc_dist(bottom_left, bottom_right)
    right = _calc_dist(bottom_right, top_right)
    top = _calc_dist(top_right, top_left)

    box = [[top_left], [bottom_left], [bottom_right], [top_right]]

    max_x = int(max([top, btm]))
    max_y = int(max([left, right]))
    pre = np.float32(box)
    post = np.float32([[0,0], [0, max_y], [max_x, max_y], [max_x, 0]])

    M = cv2.getPerspectiveTransform(pre, post)
    return max_x, max_y, M

def cut_img(img,max_x,max_y,M):
    """
    カメラで取得した画像を、他のカメラと重なりが無いようにトリミングして返す
    """
    return cv2.warpPerspective(img, M, (max_x,max_y))

def get_corner(device):
    original_zl = 645
    original_zr = 7454
    original_zm = (original_zl+original_zr)/2#中点は指定の値が与えられる可能性あり
    original_xb = 6743
    original_xt = 3176
    original_xm = (original_xb+original_xt)/2#中点は指定の値が与えられる可能性あり
    if device == 2:
        xb = original_xm
        xt = original_xt
        zr = original_zm
        zl = original_zl
    elif device == 0:
        xb = original_xm
        xt = original_xt
        zr = original_zr
        zl = original_zm
    elif device == 3:
        xb = original_xb
        xt = original_xm
        zr = original_zm
        zl = original_zl
    elif device == 1:
        xb = original_xb
        xt = original_xm
        zr = original_zr
        zl = original_zm
    else:
        print("デバイス番号は0,1,2,3のみ")

    return xb, xt, zr, zl

def to_unity_coordinate(max_x,max_y,c,device):
    """matplotlibの座標をunity座標に変換"""
    xb, xt, zr, zl = get_corner(device)
    #matplot座標の右下からの座標
    tmp_x = max_x - c[1]
    tmp_y = max_y - c[2]
    #unity座標に変換
    z = (zl - zr)*(tmp_x/max_x) + zr
    x = (xt - xb)*(tmp_y/max_y) + xb
    return [x,z]

def main(img):
    h,w,c = img.shape#hが縦(短い),wが横(長い)
    """色と位置を見つける"""
    pos_red = detect_red_color(img)#赤色のブロックを探す
    # print("Red_Coordinate:",pos_red)
    pos_blue = detect_blue_color(img)#青色のブロックを探す
    # print("Blue_Coordinate:",pos_blue)
    pos_yellow = detect_yellow_color(img)#黄色のブロックを探す
    # print("Yellow_Coordinate:",pos_yellow)
    pos_green = detect_green_color(img)
    # print("Green_Coordinate:",pos_green)
    #pos_value = detect_value(img)#明度と彩度から色付きのブロックの位置がわかる
    #print("Value_Coordinate:",pos_value)

    ans=[]

    if pos_blue is not None:
        cv2.circle(img,pos_blue,10,(0,0,255),-1)#マーク
        cv2.putText(img,'Blue',(pos_blue[0],pos_blue[1]),font,2,(255,0,255),3)
        ans.append(["Blue",pos_blue[0],pos_blue[1]])
    if pos_red is not None:
        cv2.circle(img,pos_red,10,(0,0,255),-1)#マーク
        cv2.putText(img,'Red',(pos_red[0],pos_red[1]),font,2,(255,0,255),3)
        ans.append(["Red",pos_red[0],pos_red[1]])
    if pos_yellow is not None:
        cv2.circle(img,pos_yellow,10,(0,0,255),-1)#マーク
        cv2.putText(img,'Yellow',(pos_yellow[0],pos_yellow[1]),font,2,(255,0,255),3)
        ans.append(["Yellow",pos_yellow[0],pos_yellow[1]])
    if pos_green is not None:
        cv2.circle(img,pos_green,10,(0,0,255),-1)#マーク
        cv2.putText(img,'Green',(pos_green[0],pos_green[1]),font,2,(255,0,255),3)
        ans.append(["Green",pos_green[0],pos_green[1]])

    cv2.imwrite("result.jpg",img)
    return ans

def get_img_corner(device):
    """
    デバイス番号に対応した画像における、切り取る領域を指定
    """
    if device == 0:
        top_left = [640-79, 16]
        bottom_left = [640-82, 263]
        top_right = [640-572, 17]
        bottom_right = [640-573, 260]
    elif device == 1:
        top_left = [640-65, 35]
        bottom_left = [640-58, 295]
        top_right = [640-550, 31]
        bottom_right = [640-548, 290]
    else:
        assert False, f"device {device} is not defined"
    return top_left,top_right,bottom_left,bottom_right

def run_detect(img, device=0):
    """
    実行
    args:
        - img (np.ndarray) : 切り取り前の画像
        - device (int) : デバイス番号
    """
    top_left, top_right, bottom_left, bottom_right = get_img_corner(device)
     #マップのエリアのみをトリミング
    x,y,M = _get_transform_util(top_left,top_right,bottom_left,bottom_right)
    revised_img = cut_img(img,x,y,M)
    #plt.imshow(revised_img)
    #plt.show()#トリミングが綺麗にできているか確認用
    #色付きブロックを探す旅
    color_and_pos = main(revised_img)
    # print("matplolib座標:",color_and_pos)
    out = []
    #unityの座標に変換する
    for i in range(len(color_and_pos)):
        res = to_unity_coordinate(x,y,color_and_pos[i],device)
        # print([color_and_pos[i][0],res[0],res[1]])#この結果をunityに返す
        out.append([color_and_pos[i][0],res[0],res[1]])
    return out


if __name__ == "__main__":
    #ここは適宜パスを通す
    img = cv2.imread('/Users/fukuzaki/Downloads/lego_color_detection/画像/test2_17.png')
    #デバイス番号(任意),ここではカメラは４つあるとして,左上:0,右上:1,左下:2,右下:3としている
    device_num = 0#今回は左上のカメラから画像を得たとする
    #以下のマップの四隅の値はあらかじめ設定する必要がある
    top_left = [122,109]
    bottom_left = [118,593]
    top_right = [1109,86]
    bottom_right = [1113,571]
    #マップのエリアのみをトリミング
    x,y,M = _get_transform_util(top_left,top_right,bottom_left,bottom_right)
    revised_img = cut_img(img,x,y,M)
    #plt.imshow(revised_img)
    #plt.show()#トリミングが綺麗にできているか確認用
    #色付きブロックを探す旅
    color_and_pos = main(revised_img)
    print("matplolib座標:",color_and_pos)
    #unityの座標に変換する
    for i in range(len(color_and_pos)):
        res = to_unity_coordinate(x,y,color_and_pos[i],device_num)
        print([color_and_pos[i][0],res[0],res[1]])#この結果をunityに返す
