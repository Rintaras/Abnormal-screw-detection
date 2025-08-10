import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from ultralytics import YOLO
import os
import json
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class YOLOTrainingThread(QThread):
    """YOLOモデルの学習を別スレッドで実行"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, model, data_path):
        super().__init__()
        self.model = model
        self.data_path = data_path
        
    def run(self):
        try:
            self.progress_signal.emit("YOLOモデルの学習を開始します...")
            # カスタムデータセットで学習
            self.model.train(
                data=self.data_path,
                epochs=50,
                imgsz=640,
                batch=8,
                name='neji_detection'
            )
            self.progress_signal.emit("学習が完了しました！")
            self.finished_signal.emit()
        except Exception as e:
            self.progress_signal.emit(f"学習エラー: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLOネジ検知システム')
        self.setGeometry(100, 100, 1200, 800)
        
        # メインウィジェット
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 左側：カメラ表示
        self.camera_widget = QWidget()
        camera_layout = QVBoxLayout(self.camera_widget)
        
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(800, 600)
        self.label.setStyleSheet('border: 2px solid gray;')
        camera_layout.addWidget(self.label)
        
        # 右側：コントロールパネル
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # ボタン類
        self.start_button = QPushButton('カメラ開始')
        self.start_button.clicked.connect(self.start_camera)
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton('カメラ停止')
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        self.capture_button = QPushButton('画像キャプチャ')
        self.capture_button.clicked.connect(self.capture_image)
        self.capture_button.setEnabled(False)
        control_layout.addWidget(self.capture_button)
        
        self.train_button = QPushButton('YOLO学習開始')
        self.train_button.clicked.connect(self.start_training)
        control_layout.addWidget(self.train_button)
        
        # 情報表示
        self.info_label = QLabel('システム情報')
        self.info_label.setStyleSheet('background-color: #f0f0f0; padding: 10px;')
        control_layout.addWidget(self.info_label)
        
        layout.addWidget(self.camera_widget, 2)
        layout.addWidget(control_widget, 1)
        
        # YOLOモデル初期化
        self.model = None
        self.model_path = 'best.pt'  # 学習済みモデルパス
        self.load_yolo_model()
        
        # カメラ関連
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        # 学習関連
        self.training_thread = None
        self.dataset_path = 'dataset.yaml'
        self.images_dir = 'images'
        self.labels_dir = 'labels'
        
        # 検出結果
        self.detections = []
        self.confidence_threshold = 0.5
        
        # ネジ締め具合学習用（従来版の機能）
        self.tightness_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, alpha=0.001)
        self.tightness_scaler = StandardScaler()
        self.X_train = []
        self.y_train = []
        self.is_fitted = False
        self.last_features = None
        self.last_frame = None
        self.last_prob = 0.5
        self.class_labels = np.array([0, 1])  # 0:外れ, 1:締め
        self.learn_count = 0  # 学習回数
        self.label_set = set()
        self.prob_history = []  # 確率の移動平均用
        self.prob_window = 10
        self.last_bbox = None  # バウンディングボックス情報
        
        # ディレクトリ作成
        self.create_directories()
        
        self.update_info()

    def create_directories(self):
        """必要なディレクトリを作成"""
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
    def load_yolo_model(self):
        """YOLOモデルを読み込み"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self.info_label.setText(f'モデル読み込み成功: {self.model_path}')
            else:
                # 事前学習済みモデルを使用
                self.model = YOLO('yolov8n.pt')
                self.info_label.setText('事前学習済みモデルを使用中')
        except Exception as e:
            self.info_label.setText(f'モデル読み込みエラー: {str(e)}')
            self.model = None

    def start_camera(self):
        """カメラを開始"""
        if self.cap is None:
            # カメラデバイス番号を変更可能（0:内蔵カメラ, 1:外部カメラ）
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.timer.start(30)
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.capture_button.setEnabled(True)
                self.info_label.setText('カメラ開始')
            else:
                self.info_label.setText('カメラを開けませんでした')

    def stop_camera(self):
        """カメラを停止"""
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.capture_button.setEnabled(False)
            self.info_label.setText('カメラ停止')

    def capture_image(self):
        """現在のフレームを画像として保存"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                image_path = os.path.join(self.images_dir, f'neji_{timestamp}.jpg')
                cv2.imwrite(image_path, frame)
                self.info_label.setText(f'画像保存: {image_path}')
                
                # ラベルファイルも作成（空の状態）
                label_path = os.path.join(self.labels_dir, f'neji_{timestamp}.txt')
                with open(label_path, 'w') as f:
                    pass  # 空のラベルファイル
                
                self.create_dataset_yaml()

    def create_dataset_yaml(self):
        """データセット設定ファイルを作成"""
        dataset_config = {
            'path': '.',
            'train': 'images',
            'val': 'images',
            'names': {
                0: 'neji'
            }
        }
        
        with open(self.dataset_path, 'w') as f:
            import yaml
            yaml.dump(dataset_config, f, default_flow_style=False)

    def start_training(self):
        """YOLO学習を開始"""
        if self.training_thread is None or not self.training_thread.isRunning():
            self.training_thread = YOLOTrainingThread(self.model, self.dataset_path)
            self.training_thread.progress_signal.connect(self.update_training_info)
            self.training_thread.finished_signal.connect(self.training_finished)
            self.training_thread.start()
            self.train_button.setEnabled(False)
            self.train_button.setText('学習中...')

    def update_training_info(self, message):
        """学習進捗を更新"""
        self.info_label.setText(message)

    def training_finished(self):
        """学習完了時の処理"""
        self.train_button.setEnabled(True)
        self.train_button.setText('YOLO学習開始')
        self.load_yolo_model()  # 学習済みモデルを再読み込み

    def extract_features(self, frame):
        """従来版の特徴量抽出機能"""
        h, w, _ = frame.shape
        size = min(h, w) // 3
        cx, cy = w // 2, h // 2
        crop = frame[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # ヒストグラム
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist = hist / (np.sum(hist) + 1e-6)
        # エッジ量
        edges = cv2.Canny(gray, 100, 200)
        edge_score = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)
        # 円検出
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=size//4,
                                   param1=50, param2=30, minRadius=size//6, maxRadius=size//2)
        circle_score = 1.0 if circles is not None else 0.0
        # HOG特徴量
        winSize = (gray.shape[1]//2*2, gray.shape[0]//2*2)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        resized = cv2.resize(gray, winSize)
        hog_feat = hog.compute(resized).flatten()
        # バウンディングボックス情報を保存
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # 一番大きい円を採用
            main_circle = max(circles[0, :], key=lambda c: c[2])
            # 元画像座標に変換
            x = main_circle[0] + (cx - size//2)
            y = main_circle[1] + (cy - size//2)
            r = main_circle[2]
            self.last_bbox = ('circle', (x, y, r))
        else:
            # クロップ領域を矩形として保存
            self.last_bbox = ('rect', (cx-size//2, cy-size//2, size, size))
        # 特徴量ベクトル
        features = np.concatenate([hist, [edge_score, circle_score], hog_feat])
        return features

    def fit_tightness_model(self):
        """ネジ締め具合モデルを学習"""
        # バッチ学習: 全データで再fit
        if len(self.label_set) < 2:
            return
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        self.tightness_scaler.fit(X)
        X_scaled = self.tightness_scaler.transform(X)
        self.tightness_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, alpha=0.001)
        self.tightness_model.fit(X_scaled, y)
        self.is_fitted = True

    def update_frame(self):
        """フレーム更新"""
        if self.cap is None or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        # YOLO検出
        if self.model is not None:
            results = self.model(frame, conf=self.confidence_threshold)
            self.detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        self.detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls
                        })
        
        # ネジ締め具合の特徴量抽出（従来版機能）
        features = self.extract_features(frame)
        self.last_features = features
        self.last_frame = frame
        
        # ネジ締め具合の判定
        tightness_prob = 0.5
        if self.is_fitted and len(self.label_set) == 2:
            X = self.tightness_scaler.transform([features])
            prob = self.tightness_model.predict_proba(X)[0][1]
            # 移動平均で平滑化
            self.prob_history.append(prob)
            if len(self.prob_history) > self.prob_window:
                self.prob_history.pop(0)
            tightness_prob = np.mean(self.prob_history)
            self.last_prob = tightness_prob
        else:
            self.last_prob = tightness_prob
        
        # 画像をQt形式に変換
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.label.width(), self.label.height(), Qt.KeepAspectRatio
        )
        
        # 検出結果を描画
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor('red'), 3))
        painter.setFont(QFont('Arial', 16))
        
        for detection in self.detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # 座標変換
            scale_x = pixmap.width() / w
            scale_y = pixmap.height() / h
            
            rect_x = int(x1 * scale_x)
            rect_y = int(y1 * scale_y)
            rect_w = int((x2 - x1) * scale_x)
            rect_h = int((y2 - y1) * scale_y)
            
            # バウンディングボックス描画
            painter.drawRect(rect_x, rect_y, rect_w, rect_h)
            
            # 信頼度表示
            painter.drawText(rect_x, rect_y - 10, f'Neji: {conf:.2f}')
        
        # ネジ締め具合のバウンディングボックス描画（従来版機能）
        if self.last_bbox is not None:
            painter.setPen(QPen(QColor('yellow'), 4))
            if self.last_bbox[0] == 'circle':
                x, y, r = self.last_bbox[1]
                scale_x = pixmap.width() / w
                scale_y = pixmap.height() / h
                painter.drawEllipse(int(x*scale_x - r*scale_x), int(y*scale_y - r*scale_y), int(2*r*scale_x), int(2*r*scale_y))
            elif self.last_bbox[0] == 'rect':
                x, y, size, size2 = self.last_bbox[1]
                scale_x = pixmap.width() / w
                scale_y = pixmap.height() / h
                painter.drawRect(int(x*scale_x), int(y*scale_y), int(size*scale_x), int(size2*scale_y))
        
        # テキスト描画
        painter.setPen(QColor('white'))
        painter.setFont(QFont('Arial', 24))
        tightness_percent = int(tightness_prob * 100)
        painter.drawText(20, 50, f'締め付け確率: {tightness_percent}%')
        painter.setFont(QFont('Arial', 18))
        painter.drawText(20, 80, f'学習回数: {self.learn_count}')
        painter.drawText(20, 110, f'YOLO検出数: {len(self.detections)}')
        
        painter.end()
        self.label.setPixmap(pixmap)
        
        # ウィンドウの縁色を更新（従来版機能）
        if tightness_percent >= 80:
            self.setStyleSheet('border: 8px solid green;')
        else:
            self.setStyleSheet('border: 8px solid red;')

    def keyPressEvent(self, event):
        """キーイベント処理（従来版のy/nキー機能）"""
        if self.last_features is None:
            return
        if event.key() == Qt.Key_Y:
            label = 1
        elif event.key() == Qt.Key_N:
            label = 0
        else:
            return
        self.X_train.append(self.last_features)
        self.y_train.append(label)
        self.learn_count += 1
        self.label_set.add(label)
        self.fit_tightness_model()
        self.info_label.setText(f'学習完了: ラベル={label}, 学習回数={self.learn_count}')

    def update_info(self):
        """情報表示を更新"""
        info_text = f"""
YOLOネジ検知システム

YOLO検出数: {len(self.detections)}
信頼度閾値: {self.confidence_threshold}
YOLOモデル: {'読み込み済み' if self.model else '未読み込み'}

ネジ締め具合学習:
- 学習回数: {self.learn_count}
- 締め付け確率: {int(self.last_prob * 100)}%
- モデル状態: {'学習済み' if self.is_fitted else '未学習'}

操作方法:
- カメラ開始/停止: ボタンクリック
- 画像キャプチャ: ボタンクリック
- YOLO学習: ボタンクリック
- 締め具合学習: Yキー(締め)/Nキー(外れ)
        """
        self.info_label.setText(info_text)

    def closeEvent(self, event):
        """アプリケーション終了時の処理"""
        if self.cap is not None:
            self.cap.release()
        if self.training_thread is not None and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 