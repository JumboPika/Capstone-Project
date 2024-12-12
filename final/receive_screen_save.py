import cv2
import socket
import struct
import numpy as np
import os
from datetime import datetime
import time

# 多播設置
MULTICAST_GROUP = '224.0.0.2'
MULTICAST_PORT = 5004

# 初始化套接子
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.bind(('', MULTICAST_PORT))
mreq = struct.pack('4sl', socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2**20)  # 增加接收緩衝區大小

# 影片儲存設定
SAVE_INTERVAL = 2 * 60  # 影片儲存間隔（秒）
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 儲存影片的格式
video_writer = None
record_folder = "recordings"
os.makedirs(record_folder, exist_ok=True)
start_time = time.time()
video_filename = None

# 緩衝區
buffer = b""

# 壓縮品質設置
ENCODE_QUALITY = 50  # JPEG 壓縮品質
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), ENCODE_QUALITY]

while True:
    # 接收數據包
    packet, _ = sock.recvfrom(65536)  # 調整緩衝區大小以適應串流
    buffer += packet  # 將接收到的包加入緩衝區

    # 查找 JPEG 圖像的起始和結束標記
    start_idx = buffer.find(b'\xff\xd8')  # JPEG 開始標記
    end_idx = buffer.find(b'\xff\xd9')  # JPEG 結束標記

    # 如果找到完整的 JPEG 數據
    if start_idx != -1 and end_idx != -1:
        # 截取完整的 JPEG 數據
        jpg_data = buffer[start_idx:end_idx + 2]
        buffer = buffer[end_idx + 2:]  # 剩餘數據繼續存放在緩衝區中

        # 解碼 JPEG 圖像
        frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        if frame is not None:
            # 如果沒有初始化影片寫入器，或儲存時間間隔到了
            if video_writer is None or time.time() - start_time >= SAVE_INTERVAL:
                # 關閉上一段影片
                if video_writer is not None:
                    video_writer.release()

                # 設定新檔案名稱
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = os.path.join(record_folder, f"{timestamp}.avi")
                video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                start_time = time.time()  # 更新影片開始時間

            # 壓縮影像以減少帶寬需求
            _, compressed_frame = cv2.imencode('.jpg', frame, encode_param)
            frame = cv2.imdecode(compressed_frame, cv2.IMREAD_COLOR)

            # 寫入影片
            video_writer.write(frame)

            # 將圖像改變為原始的二倍大
            resized_frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2), interpolation=cv2.INTER_LINEAR)

            # 顯示影像
            cv2.imshow('Multicast Stream', resized_frame)

            # 如果按下 ESC 鍵，退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

# 清理資源
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
sock.close()
