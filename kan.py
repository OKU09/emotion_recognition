import sounddevice as sd
import queue
import sys
import json
from vosk import Model, KaldiRecognizer
from transformers import pipeline

# Flask API 用
from flask import Flask, jsonify
import threading

# ------------------------
# Flask API サーバー
# ------------------------
app = Flask(__name__)

# 最新の感情結果を格納
latest_emotion = {}

@app.route("/emotion/latest")
def get_latest_emotion():
    return jsonify(latest_emotion)

# Flask を別スレッドで起動
api_thread = threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=5000, debug=False)
)
api_thread.daemon = True
api_thread.start()

# ------------------------
# モデルの読み込み
# ------------------------
MODEL_PATH = "C:/Users/33mok/media_exercise/vosk-model-small-ja-0.22"
try:
    model = Model(MODEL_PATH)
except Exception as e:
    print(f"モデルが見つかりません: {e}")
    sys.exit(1)

recognizer = KaldiRecognizer(model, 16000)

# rinna 日本語感情分類モデル
classifier = pipeline(
    "text-classification",
    model="C:/Users/33mok/media_exercise/kanjou-model"
)

label_map = {
    "LABEL_0": "喜び",
    "LABEL_1": "悲しみ",
    "LABEL_3": "驚き",
    "LABEL_4": "怒り",
    "LABEL_5": "恐れ",
    "LABEL_6": "嫌悪",
}

# ------------------------
# 音声データキュー
# ------------------------
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"ステータス: {status}", file=sys.stderr)
    q.put(bytes(indata))

# ------------------------
# メインループ
# ------------------------
print("マイクに向かって話してください。Ctrl+Cで終了します。")

with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback):
    try:
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    print("認識結果:", text)

                    # 感情分類
                    results = classifier(text, return_all_scores=True)[0]
                    print("感情割合:")
                    for r in results:
                        if r['label'] in ["LABEL_2", "LABEL_7"]:
                            continue
                        label_name = label_map.get(r['label'], r['label'])
                        print(f"  {label_name}: {r['score']:.2f}")

                        #  最新の感情値を Flask API に反映
                        latest_emotion[label_name] = r['score']

            else:
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "")
                if text:
                    print("途中結果:", text)

    except KeyboardInterrupt:
        print("\nプログラムを終了します。")
        sys.exit(0)
