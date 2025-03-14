import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np # 수치 계산
import pandas as pd # csv 파일 읽&저장
import serial # pyserial 라이브러리 사용
import time
from collections import deque

# 1. 데이터 수집 from 아두이노
def collect_data (serial_port="/dev/ttyUSB0", bps=9600, collect_time = 60):
    ser = serial.Serial(serial_port)
    start_time = time.time()
    data = [] # 데이터 저장 리스트

    while time.time() - start_time < collect_time :
        line = ser.readline().decode().strip()
        voltage, current = map(float, line.split(","))
        data.append([voltage, current])
        with open("log.txt", "a") as f: # 데이터셋 for 학습
            f.write(f"{voltage},{current}\n")

    ser.close()
    return np.array(data)

# 데이터 전처리
def process_data(csvfile_path, seq_length = 50) :
    df = pd.read_csv(csvfile_path)
    data = df[['voltage', 'current']].values # df -> numpy 변환

    input_data = [] # 입력 데이터 # 전류, 전압
    predict_value = [] # 예측 전류 값

    for i in range(len(data) - seq_length):
        input_data.append(data[i:i + seq_length])
        predict_value(data[i + seq_length, 1]) # 전류 예측

    return (torch.tensor(input_data, dtype=torch.float32),
            torch.tensor(predict_value, dtype = torch.float32))

# LSTM 모델 정의
class LSTMModel(nn.Module) :
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=1) :
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x) :
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# 학습 과정

# 이상현상 감지 + 아두이노에게 중지 요청

# 메인 실행
if __name__ == "__main__" :
    # 10초 동안 데이터 수집
    collected_data = collect_data(collect_time = 10)

    # 데이터 저장
    df = pd.DataFrame(collected_data, columns = ["voltage", "current"])
    df.to_csv(path_or_buf = "log.txt", index = False)