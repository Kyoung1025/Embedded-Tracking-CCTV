# server.py: YOLO 기반 추적 및 TCP 알람 신호 전송 서버

import imagezmq
import cv2
import numpy as np
import threading
import socket
from ultralytics import YOLO
import time

# ====================================================
# 1. 설정 및 상수 정의
# ====================================================

# AI 모델 및 탐지 설정
MODEL_NAME = 'yolov8s.pt'    
CONFIDENCE_THRESHOLD = 0.5  # 탐지 신뢰도 임계값 (50% 이상만 유효)
TARGET_CLASS_ID = 0         # COCO 데이터셋에서 'person' (사람)의 ID는 0
ALARM_FRAME_THRESHOLD = 60  # 사람이 연속적으로 60프레임 (약 2초) 이상 감지되어야 알람 발령

# 통신 및 추적 설정
TCP_IP = '0.0.0.0'
TCP_PORT = 6000
rpi_client_socket = None    # RPi와의 연결 소켓 (전역 변수)
person_tracking_history = {}# {track_id: 연속 감지 프레임 수} 저장
FRAME_COUNT = 0             # 현재 프레임 카운트

# ====================================================
# 2. TCP 알람 서버 함수 정의
# ====================================================

def tcp_alarm_server():
    """
    라즈베리파이가 접속할 포트(6000)를 열고 연결을 기다립니다.
    연결이 성공하면 rpi_client_socket 변수에 소켓 객체를 저장합니다.
    """
    global rpi_client_socket
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.bind((TCP_IP, TCP_PORT))
        server_socket.listen(1)
        print(f"[TCP Server] Alarm Server listening on port {TCP_PORT}...")
        
        rpi_client_socket, addr = server_socket.accept()
        print(f"[TCP Server] !!! RPi Client connected from {addr} !!!")
        
    except Exception as e:
        print(f"[TCP Server ERROR] Error starting TCP server: {e}")
    finally:
        server_socket.close()


# ====================================================
# 3. 메인 프로그램 시작
# ====================================================

if __name__ == '__main__':
    
    # 3.1. YOLO 모델 로드
    try:
        model = YOLO(MODEL_NAME)
        print(f"[YOLO] Model ({MODEL_NAME}) loaded successfully.")
    except Exception as e:
        print(f"[YOLO ERROR] Error loading YOLO model: {e}")
        exit()

    # 3.2. TCP 알람 서버 스레드 실행
    tcp_thread = threading.Thread(target=tcp_alarm_server)
    tcp_thread.daemon = True
    tcp_thread.start()

    # 3.3. ImageZMQ 수신 시작
    receiver = imagezmq.ImageHub(open_port='tcp://*:5555')
    print("[ImageZMQ] Receiver Server is running. Waiting for RPi video stream...")

    while True:
        FRAME_COUNT += 1
        
        try:
            # RPi로부터 영상 프레임 수신
            frame_name, frame = receiver.recv_image()
        except Exception as e:
            print(f"[ImageZMQ ERROR] Failed to receive frame: {e}")
            break

        is_intruder_confirmed = False
        current_frame_person_ids = set()
        
        # 3.4. YOLO 추적 실행 (SORT 역할)
        # tracker="bytetrack.yaml"을 사용하여 안정적인 객체 ID 추적 활성화
        results = model.track(frame, conf=CONFIDENCE_THRESHOLD, 
                              verbose=False, persist=True, tracker="bytetrack.yaml")

        # 3.5. 결과 분석 및 추적 로직 업데이트
        for r in results:
            
            # 추적된 객체(id)가 있어야만 처리합니다.
            if r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int) 
                classes = r.boxes.cls.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                
                for box, cls_id, track_id in zip(boxes, classes, track_ids):
                    
                    # 'person' (0) 객체만 필터링
                    if int(cls_id) == TARGET_CLASS_ID:
                        
                        current_frame_person_ids.add(track_id)
                        
                        # 추적 기록 업데이트 (감지 프레임 수 증가)
                        person_tracking_history[track_id] = person_tracking_history.get(track_id, 0) + 1
                        
                        # 침입 확정 판단
                        if person_tracking_history[track_id] >= ALARM_FRAME_THRESHOLD:
                            is_intruder_confirmed = True
                        
                        # 화면에 ID와 추적 횟수 표시 (OpenCV 활용)
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'ID:{track_id} F:{person_tracking_history[track_id]}', 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 3.6. 추적 종료된 ID 정리 (안정성 강화)
        # 현재 프레임에 잡히지 않은 ID의 카운트를 감소시켜야 오탐지를 줄일 수 있습니다.
        keys_to_decrement = [id for id in person_tracking_history if id not in current_frame_person_ids]
        for id in keys_to_decrement:
            # 10프레임의 여유를 줍니다.
            person_tracking_history[id] = max(0, person_tracking_history[id] - 10) 
            if person_tracking_history[id] == 0:
                del person_tracking_history[id] # 0이 되면 기록에서 삭제

        # 3.7. 알람 신호 전송
        if is_intruder_confirmed and rpi_client_socket:
            try:
                # 알람 신호 전송 (줄바꿈 문자를 포함하여 RPi가 메시지 구분을 쉽게 하도록 함)
                rpi_client_socket.sendall("ALARM_PERSON\n".encode('utf-8'))
                cv2.putText(frame, '!!! INTRUDER CONFIRMED & ALARM SENT !!!', (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            except Exception as e:
                print(f"[TCP Send ERROR] Error sending alarm signal: {e}. RPi disconnected.")
                rpi_client_socket = None # 연결이 끊어졌으므로 소켓 초기화
                
        # 3.8. 화면 표시 및 루프 제어
        cv2.imshow("Smart CCTV Stream (Server)", frame)
        receiver.send_reply(b'OK') 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Server stopped.")