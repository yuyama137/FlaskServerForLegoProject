from flask import Flask, request, jsonify #Flaskクラスのインポート
import base64
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
from PIL import Image
import time
from detect import run_detect
import util

app = Flask(__name__) #appという名前のFlaskクラスのインスタンスを作成

@app.route("/", methods=["POST", "GET"]) #ルーティング(URLの設定）
def hello(): #"/hello"のURLで呼び出される関数
    print("aac")
    #print(request)
    R = request.get_json(force=True)
    print(type(R))
    print(f"device num : {R['device']}")
    #print(R["img"])
    # b64encoded = base64.b64encode(R["img"]).decode("latin-1")
    b64_bytes = R["img"].encode()
    # count = R["cnt"]
    # print(base64.decodebytes(b64_bytes))
    b64_img = base64.decodebytes(b64_bytes)

    image = Image.open(io.BytesIO(b64_img))
    image_np = np.array(image.convert("RGB"))
    image_np_bgr = cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)

    # time.sleep(4);
    print("end python")
    cv2.imwrite(f"capture\\device{R['device']}_{time.time()}.png", image_np_bgr)

    out_data = run_detect(image_np_bgr, R["device"])

    res = util.list_to_json_dic(out_data)

    res["message"] = "got at python"
    # 座標群 : [1.1,2.2],[3.3,4.4],[5.5,6.6] 
    # length = np.random.randint(4,10)
    # res["x"] = np.random.randint(0,20,length).tolist()
    # res["z"] = np.random.randint(0,20,length).tolist()
    res["device"] = R["device"]
    
    # # debug
    # res["x"] = [float(4193)]
    # res["z"] = [float(6535)]
    return jsonify(res)




    # return R
    # curl -H "Content-type: application/json" -X POST -d "{\"name\":\"aiueo\"}"  http://localhost:5000/   
    # jsonの中にエスケープキーが必要な事に注意

if __name__ == "__main__":
    app.run(debug=True) #Webサーバーを立ち上げる