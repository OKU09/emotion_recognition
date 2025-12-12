import requests
import time

# 3つのサーバーのURL
urls = [
    "http://localhost:5000/emotion/latest",
    "http://localhost:5001/emotion/latest",
    "http://localhost:5002/emotion/latest"
]

# 各サーバーに対する重み
weights = [0.5, 0.3, 0.2]

while True:
    try:
        all_res = []
        for url in urls:
            res = requests.get(url).json()  # 例: {"happy":0.8, "sad":0.1, "angry":0.1}
            all_res.append(res)

        # ラベルをすべて集めて重み付き平均を計算
        avg_res = {}
        if all_res:
            # まず全てのラベルを集める（不揃いの場合も対応）
            all_labels = set()
            for r in all_res:
                all_labels.update(r.keys())

            for label in all_labels:
                weighted_sum = 0
                total_weight = 0
                for r, w in zip(all_res, weights):
                    if label in r:
                        try:
                            score = float(r[label]) 
                            
                            weighted_sum += score * w
                            total_weight += w
                        except ValueError:
                            # 数値に変換できない無効な値はスキップ
                            print(f"警告: ラベル '{label}' の値 '{r[label]}' は無効な数値です。スキップします。")
                            continue
                if total_weight > 0:
                    avg_res[label] = weighted_sum / total_weight
                else:
                    avg_res[label] = 0  # データなしの場合は0にする

            print("ラベルごとの加重平均:", avg_res)

    except Exception as e:
        print("取得失敗:", e)

    time.sleep(0.5)
