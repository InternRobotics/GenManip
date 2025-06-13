import socket
import pickle
import struct


def send_message(send_socket: socket.socket, data: dict):
    serialized_data = pickle.dumps(data)
    message_size = struct.pack("Q", len(serialized_data))
    send_socket.sendall(message_size + serialized_data)


def wait_message(conn: socket.socket):
    data = b""
    payload_size = struct.calcsize("Q")
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += conn.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    received_data = pickle.loads(frame_data)

    return received_data


def create_send_port_and_wait(port: int):
    serial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serial.bind(("localhost", port))
    serial.listen(1)
    print("Waiting for a connection...")
    conn, addr = serial.accept()
    print("Connected by", addr)
    return conn


def create_receive_port_and_attach(port: int):
    serial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serial.connect(("localhost", port))
    print("connected port ", port)
    return serial
