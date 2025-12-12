import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from wighted_fusion import get_weighted_emotion

# ===========================
# Windows emoji font
# ===========================
FONT_PATH = "C:/Windows/Fonts/seguiemj.ttf"
FONT = ImageFont.truetype(FONT_PATH, 80)


# ===========================
# エフェクト管理クラス
# ===========================
class EmotionEffect:
    def __init__(self):
        self.active_emotion = None
        self.timer = 0
        self.duration = 30  # 表示持続フレーム

        # happiness animation
        self.heart_offset = 0
        self.heart_frame = 0

        # fear animation
        self.fear_shift = 0
        self.fear_step = 1

    def update(self, emotion):
        # 感情が更新されたらリセット
        if emotion:
            if self.active_emotion != emotion:
                self.active_emotion = emotion
                self.timer = self.duration
                self.heart_frame = 0
                self.heart_offset = 0
                self.fear_shift = 0
        else:
            if self.timer > 0:
                self.timer -= 1
            if self.timer == 0:
                self.active_emotion = None

    def get_active(self):
        return self.active_emotion if self.timer > 0 else None


# ===========================
# ハートアニメ（未使用：必要時に利用可能）
# ===========================
def draw_small_heart(frame, x, y, frame_count):
    size = 10
    opacity = max(0, 1 - frame_count / 30)  # フェードアウト
    dy = -frame_count * 2  # 上方向へ移動

    t = np.linspace(0, 2 * np.pi, 200)
    X = size * 16 * np.sin(t)**3 + x
    Y = -size * (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)) + y + dy

    pts = np.vstack([X, Y]).T.astype(np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (255, 182, 193))
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)


# ===========================
# Pillowで絵文字描画
# ===========================
def draw_emoji(frame, text, x, y, color=None):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, font=FONT, fill=color if color else (255,255,255))
    return np.array(img_pil)


# ===========================
# メイン処理
# ===========================
def main():
    effect = EmotionEffect()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが開けません")


    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # ---------------------------
        # ダミーの感情推定（あなたのAIに置き換わる）
        # ---------------------------


        emotions = get_weighted_emotion()

        if emotions:
            main_emotion = max(emotions, key=emotions.get)
            if emotions[main_emotion] < 0.15:
                main_emotion = None
        else:
            main_emotion = None

        effect.update(main_emotion)
        emo = effect.get_active()

        
        # ===========================
        # エフェクト描画
        # ===========================
        for (x, y, fw, fh) in faces:
            right_top = (x + fw, y)
            left_top = (x, y)
            center_top = (x + fw//2, y - 20)
            right_top2 = (x + fw + 10, y - 10)

            # --- Anger（怒り） ---
            if emo == "anger":
                frame = draw_emoji(frame, "💢", right_top[0], right_top[1] - 40, color=(0,0,255))

            # --- Disgust（嫌悪） ---
            elif emo == "disgust":
                # 緑の抽象もやもや
                for i in range(20):
                    yy = y - 40 + i * 4
                    for t in range(80):
                        xx = x - 60 + int(20 * np.sin(t / 5))
                        if 0 <= yy < h and 0 <= xx < w:
                            frame[yy, xx] = (0, 180, 0)

            # --- Fear（恐怖） ---
            elif emo == "fear":
                effect.fear_shift += effect.fear_step
                if abs(effect.fear_shift) > 3:
                    effect.fear_step *= -1
                shift = effect.fear_shift

                # 左上ギザギザ
                for i in range(5):
                    cv2.line(
                        frame,
                        (left_top[0] + shift - 20, left_top[1] - 20 + i * 12),
                        (left_top[0] + shift,       left_top[1] - 10 + i * 12),
                        (255, 0, 0), 2
                    )

                # 右上ギザギザ
                for i in range(5):
                    cv2.line(
                        frame,
                        (right_top[0] + shift,        right_top[1] - 20 + i * 12),
                        (right_top[0] + 20 + shift,   right_top[1] - 10 + i * 12),
                        (255, 0, 0), 2
                    )

            # --- Happiness（幸福） ---
            elif emo == "happiness":
                heart_x = x + fw + 10
                heart_y = y - 10
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((heart_x, heart_y), "💗", font=FONT, fill=(255, 150, 180))
                frame = np.array(img_pil)

            # --- Sadness（悲しみ） ---
            elif emo == "sadness":
                frame = draw_emoji(frame, "💧", right_top2[0], right_top2[1], color=(255,0,0))

            # --- Surprise（驚き） ---
            elif emo == "surprise":
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((center_top[0] + 10, center_top[1] - 40), "!", fill=(0,0,255), font=FONT)
                frame = np.array(img_pil)

        cv2.imshow("Emotion Effect", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ===========================
# エントリーポイント
# ===========================
if __name__ == "__main__":
    main()
