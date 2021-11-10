import numpy as np


def list_to_json_dic(lst, device=0):
    """
    unityに返すjosnに対応する形の辞書に変換する。
    [[color1, x1, z1], [color2, x2, z2], [color3, x3, z3], ...] 
            => {
                'color' : [color1, color2, color3, ...],
                'x' : [x1, x2, x3, ...], 
                'y' : [y1, y2, y3, ...],
                'device' : 0
            }
    """
    out = {}
    out["device"] = device
    out["color"] = []
    out["x"] = []
    out["z"] = []
    for d in lst:
        out["color"].append(d[0])
        out["x"].append(d[1])
        out["z"].append(d[2])
    return out






