from dataclasses import dataclass, field
from collections import deque
import threading
import pickle


@dataclass
class Buffer:
    data_list: deque = field(default_factory=deque)
    buffer_length: int = 50
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, data):
        with self.lock:
            if len(self.data_list) >= self.buffer_length:
                self.data_list.popleft()
            self.data_list.append(data.copy())

    def get_data(self):
        with self.lock:
            return list(self.data_list)

    def save_to_file(self, file_path: str):
        with self.lock:
            data_dict = {
                'data': list(self.data_list),
                'length': len(self.data_list)
            }
            with open(file_path, 'wb') as f:
                pickle.dump(data_dict, f)
            self.data_list.clear()  # Clear the buffer after saving
    
    def clear(self):
        self.data_list.clear()  # Clear the buffer after saving
    
    def load_from_file(self, file_path: str):
        with self.lock:
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
                self.data_list = deque(data_dict['data'])
    
    def is_full(self):
        if len(self.data_list) == self.buffer_length:
            return True
        else:
            return False
    
    def __len__(self):
        with self.lock:
            return len(self.data_list)

@dataclass
class GridDataBuffer(Buffer):
    def push(self, grid_data):
        super().push(grid_data)

@dataclass
class VehicleStateBuffer(Buffer):
    def push(self, vehicle_state):
        super().push(vehicle_state)