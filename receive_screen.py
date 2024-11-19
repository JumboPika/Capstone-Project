import cv2
import socket
import struct
import numpy as np

# 多播設定
MULTICAST_GROUP = '224.0.0.2'
MULTICAST_PORT = 5004

# 初始化套接字
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.bind(('', MULTICAST_PORT))
mreq = struct.pack('4sl', socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

buffer = b""
while True:
    # 接收資料
    packet, _ = sock.recvfrom(65536)  # 65536 是最大 UDP 封包長度
    buffer += packet

    # 嘗試解碼 JPEG 影像
    start_idx = buffer.find(b'\xff\xd8')  # JPEG 起始標誌
    end_idx = buffer.find(b'\xff\xd9')    # JPEG 結束標誌
    if start_idx != -1 and end_idx != -1:
        jpg_data = buffer[start_idx:end_idx+2]
        buffer = buffer[end_idx+2:]

        # 解碼影像並顯示
        frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            cv2.imshow('Multicast Stream', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
                break

cv2.destroyAllWindows()
sock.close()
