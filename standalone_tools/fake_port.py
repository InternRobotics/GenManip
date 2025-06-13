import argparse
import pickle
import struct
import socket
import time
import numpy as np


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


def generate_action(data, agent):
    obs = {}
    obs["robot"] = {"qpos": data["joint_position_state"]}
    obs["obs_camera"] = {"color_image": data["obs_camera_rgb"]}
    obs["realsense"] = {"color_image": data["realsense_rgb"]}
    goal = data["instruction"]
    timestep = data["timestep"]
    reset = data["reset"]
    if reset:
        agent.reset()
    output, gripper, _ = agent.forward(obs, goal, timestep)
    result = np.array(output)
    return result, gripper


def process_data(data, agent):
    try:
        processed_data, gripper = generate_action(data, agent)
        # print(processed_data, gripper)
        if processed_data is None:
            return {"message": "No action generated!"}
        action = processed_data.tolist() + (
            [0.04, 0.04] if gripper == -1 else [0.0, 0.0]
        )
        return {"action": action}
    except Exception as e:
        print(str(e))
        return {"error": str(e)}


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--receive_port", type=int, default=10012)
parser.add_argument("-s", "--send_port", type=int, default=10013)
args = parser.parse_args()

if __name__ == "__main__":
    send_socket = create_send_port_and_wait(port=args.send_port)
    time.sleep(1)
    receive_socket = create_receive_port_and_attach(port=args.receive_port)

    while True:
        data = wait_message(receive_socket)
        actions = {"action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        actions = {"action": ([0.4, 0.2, 0.3], [0.0, 1.0, 0.0, 0.0], [0.04, 0.04])}
        send_message(send_socket, actions)
