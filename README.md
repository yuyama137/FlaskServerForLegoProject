# Flaskサーバー

unityで画像をPOSTして座標をレスポンスする。

## requirements

- flask
- opencv
- numpy
- matplotlib
- PIL

## usege

```bash
python run.py
curl -H "Content-type: application/json" -X POST -d "json data"  http://localhost:5000/
```

POST Data (json)

```json
{
    "device" : "number of device (int)",
    "img" : "base64 of image",
}
```

RESPONSE Data (json)

```json
{
    "color" : "[color1, color2, color3, ...]",
    "x" : "[x1, x2, x3, ...]", 
    "y" : "[y1, y2, y3, ...]",
    "device" : "number of device (int)"
}
```



