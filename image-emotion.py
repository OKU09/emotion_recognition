import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QComboBox, QSizePolicy, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from feat import Detector
import cv2
import numpy as np
from PIL import Image
import tempfile
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.impute import SimpleImputer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.ticker as ticker
import pygetwindow as gw
import warnings
import time
import queue
import win32gui
import win32ui
import win32process
import ctypes
from ctypes import windll
from flask import Flask, jsonify
import threading


# MS Gothicフォントを設定
plt.rcParams['font.family'] = fm.FontProperties(fname='C:/Windows/Fonts/msgothic.ttc').get_name()
import locale

# システムのデフォルトエンコーディングをUTF-8に設定
if sys.platform.startswith('win'):
    try:
        locale.setlocale(locale.LC_ALL, 'Japanese_Japan.utf8')
    except locale.Error:
        pass
    try:
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
        sys._enablelegacywindowsfsencoding = False
    except Exception as e:
        print(f"エンコーディング設定エラー: {str(e)}")
        
# FutureWarningを無視する
warnings.simplefilter("ignore", category=FutureWarning)
        
class CameraProcessingThread(QThread):
    frame_ready_signal = pyqtSignal(np.ndarray)

    def __init__(self, target_window_name, frame_queue, status_label, camera_detector):
        super().__init__()
        self.running = False
        self.frame_queue = frame_queue
        self.status_label = status_label
        self.camera_detector = camera_detector
        self.facebox_data = []
    
    def get_foreground_window(self, max_retries=10, retry_delay=1.0):
        """ 現在フォーカスされているウィンドウを取得（取得できるまでリトライ） """
        retries = 0
        current_pid = win32process.GetCurrentProcessId()  # 現在のプロセスIDを取得

        while retries < max_retries:
            hwnd = win32gui.GetForegroundWindow()  # フォアグラウンドのウィンドウを取得
            if not hwnd:
                print("フォアグラウンドウィンドウの取得に失敗しました")
                time.sleep(retry_delay)
                retries += 1
                continue

            _, pid = win32process.GetWindowThreadProcessId(hwnd)

            # 自分自身のウィンドウならスキップ
            if pid == current_pid:
                retries += 1
                time.sleep(retry_delay)
                continue

            # ウィンドウ名を取得して更新
            new_window_name = win32gui.GetWindowText(hwnd)
            if new_window_name:  # 空のウィンドウ名は無視
                self.target_window_name = new_window_name  # 更新
                return hwnd

            time.sleep(retry_delay)
            retries += 1

        print("有効なフォアグラウンドウィンドウが取得できませんでした。")
        return None

    def find_window_by_partial_name(self, partial_name):
        """ 部分一致でウィンドウを検索 """
        win_list = []

        def enum_windows(hwnd, _):
            win_list.append((hwnd, win32gui.GetWindowText(hwnd)))

        win32gui.EnumWindows(enum_windows, None)

        for hwnd, title in win_list:
            if partial_name.lower() in title.lower():
                self.target_window_name = title  # 更新
                return hwnd
        return 0

    def capture_window(self, window_name):
        """ ウィンドウ名を元にスクリーンショットを取得 """
        hwnd = win32gui.FindWindow(None, window_name)

        if not hwnd:
            hwnd = self.find_window_by_partial_name(window_name)

        if not hwnd:
            hwnd = self.get_foreground_window()

        if hwnd is None:
            print("ウィンドウを取得できませんでした")
            return None  # キャプチャを実行せず終了    

        self.status_label.setText(f"ウィンドウ '{window_name}' をキャプチャ中...")
        # ウィンドウの境界を取得
        rect = ctypes.wintypes.RECT()
        windll.dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect))
        
        left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
        width, height = right - left, bottom - top

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)  # `2` を指定して全体をキャプチャ
        if not result:
            print("PrintWindow に失敗しました")
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)
            return None  # キャプチャ失敗時は `None` を返す

        # 画像データを取得
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))

        # リソース解放
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        # PIL Image オブジェクトを作成
        pil_image = Image.fromarray(img)

        # アスペクト比を保持してリサイズ
        max_width = 400
        max_height = 400
        original_width, original_height = pil_image.size
        aspect_ratio = original_width / original_height

        if original_width <= max_width and original_height <= max_height:
            # 元の画像が既に最大サイズ以下の場合はリサイズしない
            return pil_image

        # 最大サイズ以下にならない範囲で最小にリサイズ
        if original_width / max_width > original_height / max_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)

        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return pil_image

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)  # ← カメラ入力（0: 内蔵 or 最初のWebカメラ）

        if not cap.isOpened():
            print("カメラを開けませんでした。")
            return

        self.status_label.setText("カメラ入力中...")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # 最新フレームだけ保持
            if self.frame_queue.full():
                _ = self.frame_queue.get()
            self.frame_queue.put(frame)

            # 顔検出（py-feat）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            single_face_predictions = self.camera_detector.detect_faces(pil_image)

            if single_face_predictions and len(single_face_predictions) > 0:
                try:
                    single_face_prediction = single_face_predictions[0][0]
                    self.facebox_data.append(single_face_prediction)
                except (IndexError, TypeError):
                    pass

            # GUIに送信
            self.frame_ready_signal.emit(frame)
            time.sleep(0.1)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def get_facebox_data(self):
        return self.facebox_data     

class AnalysisProcessingThread(QThread):
    result_ready_signal = pyqtSignal(np.ndarray)

    def __init__(self, frame_queue):
        super().__init__()
        self.running = False
        self.emotion_data = []
        self.frame_queue = frame_queue  # カメラスレッドとのキューを共有
        self.is_processing = False  # 処理中フラグ

    def run(self):
        self.analysis_detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="xgb",
            emotion_model="resmasknet",
            # 一部変更
            #face_detection_threshold=0.95,
            face_detection_threshold=0.99, # 検出確度を上げ、ノイズ（誤検出）を減らす
            pose_model="img2pose",
        )

        self.running = True
        while self.running:
            # キューからフレームを取得
            try:
                frame_rgb = self.frame_queue.get_nowait()
            except queue.Empty:
                continue

            # 処理中であれば新しいフレームをスキップ
            if self.is_processing:
                continue

            self.is_processing = True  # 処理を開始

            try:
                # RGBAからRGBに変換
                pil_image = Image.fromarray(frame_rgb).convert('RGB')                

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    pil_image.save(temp_file.name)
                    temp_path = temp_file.name

                    single_face_prediction = self.analysis_detector.detect_image(temp_path, outputFname="output.csv")

                    if single_face_prediction.emotions.empty:
                        print("認識データがありません。次のフレームに進みます。")
                        continue

                    if single_face_prediction.emotions.isna().all(axis=None):
                        print("全ての結果がNaNのためスキップします。")
                        continue

                    print("Facebox:", single_face_prediction.faceboxes)
                    print("AUs:", single_face_prediction.aus)
                    print("Emotions:", single_face_prediction.emotions)
                    print("Facepose:", single_face_prediction.poses)

                    self.emotion_data.append(single_face_prediction.emotions)

                    figs = single_face_prediction.plot_detections(faces="aus")
                    figs[0].canvas.draw()

                    image = np.array(figs[0].canvas.renderer.buffer_rgba())
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

                    self.result_ready_signal.emit(image_bgr)

            except Exception as e:
                print(f"解析中にエラーが発生しました: {e}")

            finally:
                self.is_processing = False  # 処理が終了したらフラグを解除    

    def stop(self):
        self.running = False
        self.is_processing = False
        self.wait()

    def get_emotion_data(self):
        return self.emotion_data
   
class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        self.camera_detector = Detector(
            face_model="retinaface",
            landmark_model="mobilefacenet",
            au_model="xgb",
            emotion_model="resmasknet",
            # 一部変更
            #face_detection_threshold=0.95,
            face_detection_threshold=0.99, # 検出確度を上げ、ノイズ（誤検出）を減らす
            pose_model="img2pose",
        )

        self.setWindowTitle("画面キャプチャ表情分析")
        self.setGeometry(100, 100, 1200, 900)  # メイン画面のサイズを大きく設定

        # ウィンドウサイズを固定
        self.setFixedWidth(1200)  # 横幅1200ピクセルに固定
        self.setFixedHeight(900)  # 高さ900ピクセルに固定

        main_layout = QVBoxLayout()

        self.status_label = QLabel("キャプチャが停止しています", self)
        main_layout.addWidget(self.status_label)

        # ウィンドウ選択エリア
        window_selection_layout = QHBoxLayout()
        self.window_selector = QComboBox(self)
        self.update_window_list()
        self.window_selector.setFixedWidth(700)  # 幅を700ピクセルに設定
        window_selection_layout.addWidget(self.window_selector)

        # 「ウィンドウを再取得」ボタン
        self.refresh_button = QPushButton("ウィンドウを再取得", self)
        self.refresh_button.clicked.connect(self.update_window_list)
        window_selection_layout.addWidget(self.refresh_button)

        main_layout.addLayout(window_selection_layout)

        self.start_button = QPushButton("開始", self)
        self.start_button.clicked.connect(self.start_camera)
        main_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("終了", self)
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        main_layout.addWidget(self.stop_button)

        top_layout = QHBoxLayout()

        self.camera_image_label = QLabel(self)
        self.camera_image_label.setAlignment(Qt.AlignCenter)  # キャプチャ画像を中央に配置
        self.camera_image_label.setMaximumSize(int(320 * 1.3), int(240 * 1.3))  # 最大サイズを設定
        self.camera_image_label.setScaledContents(True)  # キャプチャ画像が枠に収まるように縮小表示
        # キャプチャ画像エリアをレイアウトに合わせてサイズを柔軟に変更
        self.camera_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.result_image_label = QLabel(self)
        self.result_image_label.setAlignment(Qt.AlignCenter)  # 分析結果を中央に配置
        self.result_image_label.setMinimumHeight(400)  # 最小高さを設定
        self.result_image_label.setScaledContents(True)  # 分析結果が枠に収まるように縮小表示
        # 分析結果エリアをレイアウトに合わせてサイズを柔軟に変更
        self.result_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        top_layout.addWidget(self.camera_image_label)
        top_layout.addWidget(self.result_image_label)
        main_layout.addLayout(top_layout)
       
        # グラフ表示を広げる
        self.fig, self.ax = plt.subplots(figsize=(12, 8))  # グラフ表示のサイズを広げる
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        self.cursor_label = QLabel(self)  # カーソル位置の値を表示するラベル
        main_layout.addWidget(self.cursor_label)

        # グリッド線の間隔を設定（Y軸は0.1単位でグリッド線を維持）
        self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # グリッド線用（0.1単位）
        self.ax.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.5, 1.0]))  # ラベル用（0.0, 0.5, 1.0 のみ）

        # X軸も1単位でグリッド線を表示
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

        # グリッド線を描画
        self.ax.grid(visible=True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # マウスイベント設定
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        self.setLayout(main_layout)

        self.capture_thread = None
        self.analysis_thread = None
        self.emotion_values = {
            'anger': np.array([]),
            'disgust': np.array([]),
            'fear': np.array([]),
            'happiness': np.array([]),
            'sadness': np.array([]),
            'surprise': np.array([]),
            'neutral': np.array([]),
        }

        self.max_plot_size = 30
        self.total_data_count = 0
        self.lines = {}

        self.api = Flask(__name__)
        self.latest_emotion = {}

        @self.api.route("/emotion/latest")
        def get_latest_emotion():
            return jsonify(self.latest_emotion)

        api_thread = threading.Thread(
        target=lambda: self.api.run(host="0.0.0.0", port=5000, debug=False)
        )
        api_thread.setDaemon(True)
        api_thread.start()


    def update_window_list(self):
        """現在のウィンドウ一覧を取得してコンボボックスに設定"""
        self.window_selector.clear()  # 既存の項目をクリア

        # 全てのウィンドウタイトルを取得
        all_windows = [title for title in gw.getAllTitles() if title.strip()]

        # フィルター条件: 複数のキーワードを指定
        keywords = ["Chrome", "Edge", "Skype", "Zoom", "Youtube"]  # 検索したいキーワードをリストで指定

        # ウィンドウタイトルがキーワードのいずれかを含む場合にフィルタリング
        filtered_windows = [
            title for title in all_windows if any(keyword in title for keyword in keywords)
        ]

        # フィルターされたタイトルをコンボボックスに追加
        self.window_selector.addItems(filtered_windows)

        # フィルター結果が空の場合、デフォルトメッセージを表示
        if not filtered_windows:
            self.status_label.setText(f"指定されたキーワード {keywords} を含むウィンドウが見つかりませんでした")
        else:
            self.status_label.setText(f"ウィンドウ一覧を更新しました ({len(filtered_windows)} 件)")

    def on_mouse_move(self, event):
        if event.inaxes == self.ax:  # マウスがグラフ内にある場合
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.cursor_label.setText(f"X: {x:.1f}, Y: {y:.2f}")  # ラベルに値を表示
            else:
                self.cursor_label.clear()  # 値をクリア
        else:
            self.cursor_label.clear()
            
    def start_camera(self):

        self.status_label.setText("カメラを起動中...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.refresh_button.setEnabled(False)

        self.frame_queue = queue.Queue(maxsize=1)

        # target_window_name の代わりにダミー値を渡す（使わないため）
        self.capture_thread = CameraProcessingThread("", self.frame_queue, self.status_label, self.camera_detector)
        self.analysis_thread = AnalysisProcessingThread(self.frame_queue)

        self.capture_thread.start()
        QTimer.singleShot(1000, self.analysis_thread.start)
        self.capture_thread.frame_ready_signal.connect(self.display_frame)
        self.analysis_thread.result_ready_signal.connect(self.display_result)


    def stop_camera(self):
        if self.capture_thread:
            self.capture_thread.stop()

        if self.analysis_thread:    
            self.analysis_thread.stop()

        self.status_label.setText("キャプチャが停止しました")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.refresh_button.setEnabled(True)  # 再取得ボタンを有効化

    def display_frame(self, frame):
        # 元の画像サイズを取得
        original_height, original_width = frame.shape[:2]

        # ビデオ画像のサイズを2倍にリサイズ
        small_frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        resized_height, resized_width = small_frame.shape[:2]

        # スケールファクターを計算
        scale_x = resized_width / original_width
        scale_y = resized_height / original_height

        # 顔認識結果を取得
        facebox_data = self.capture_thread.get_facebox_data()

        if facebox_data:
            # 最新の顔認識結果
            latest_data = facebox_data[-1]

            # latest_data はリストであり、最低4つの要素があることをチェック
            if isinstance(latest_data, list) and len(latest_data) >= 4:
                try:
                    # 元の座標を取得
                    x_min = int(latest_data[0] * scale_x)
                    y_min = int(latest_data[1] * scale_y)
                    x_max = int(latest_data[2] * scale_x)
                    y_max = int(latest_data[3] * scale_y)

                    # 矩形の描画
                    cv2.rectangle(small_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)  # 緑色の矩形
                except (IndexError, TypeError, ValueError) as e:
                    print(f"Error processing facebox data: {e}")
            else:
                print("latest_data is invalid or does not contain enough elements")

        # BGRからRGBへの変換を適用（ここで描画結果を含む small_frame を使用）
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # QImage と QPixmap を使用して GUI に表示
        height, width, channel = frame_rgb.shape
        bytes_per_line = channel * width
        qimg = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        self.camera_image_label.setPixmap(pixmap)

        # GUI の再描画
        self.camera_image_label.repaint()
        self.camera_image_label.setScaledContents(True)

    def display_result(self, result):
        """リアルタイムの分析結果をウィジェットに表示"""
        if isinstance(result, np.ndarray):
            height, width, channel = result.shape
            bytes_per_line = channel * width
            qimg = QImage(result.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimg)
            self.result_image_label.setPixmap(pixmap)

            self.result_image_label.repaint()
            self.result_image_label.setScaledContents(True)

            # グラフをリアルタイムで更新
            self.update_emotion_plot()

    def update_emotion_plot(self):
        # Analysis_threadの存在を確認
        if self.analysis_thread is None:
            print("Error: AnalysisThread is not initialized")
            return

        # 新しいデータを取得
        try:
            emotion_data = self.analysis_thread.get_emotion_data()
        except AttributeError as e:
            print(f"Error while accessing get_emotion_data: {e}")
            return

        if not emotion_data:
            print("No emotion data available")
            return

        # データの更新処理
        for data in emotion_data[-1:]:
            for emotion in self.emotion_values:
                value = data[emotion].values[0] if emotion in data.columns else np.nan
                self.emotion_values[emotion] = np.append(self.emotion_values[emotion], value)

            self.total_data_count += 1

        # 欠損値を補完
        imputer = SimpleImputer(strategy="mean")
        for emotion in self.emotion_values:
            self.emotion_values[emotion] = imputer.fit_transform(self.emotion_values[emotion].reshape(-1, 1)).flatten()

        # 初期化されていない場合にラインを作成
        if not self.lines:
            for emotion in self.emotion_values:
                line, = self.ax.plot([], [], label=emotion)
                self.lines[emotion] = line

        # データをプロット
        for emotion in self.emotion_values:
            plot_data = self.emotion_values[emotion][-self.max_plot_size:]
            self.lines[emotion].set_data(
                range(self.total_data_count - len(plot_data) + 1, self.total_data_count + 1),
                plot_data
            )

        # グラフの範囲とラベルを設定
        self.ax.set_xlim(max(1, self.total_data_count - self.max_plot_size + 1), self.total_data_count)
        self.ax.set_ylim(0, 1.0)
        self.ax.set_title("Emotion Change Over Time", fontsize=14)
        self.ax.set_xlabel("Frames", fontsize=10)
        self.ax.set_ylabel("Emotion Score", fontsize=10)
        self.ax.legend(fontsize=8)

        # レイアウトと再描画
        self.fig.tight_layout()
        self.canvas.draw()

        self.latest_emotion = {
            emotion: float(self.emotion_values[emotion][-1])
            for emotion in self.emotion_values
        }

def main():
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
