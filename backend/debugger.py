import socket
from contextlib import closing
from os import getenv


def is_port_open(host, port):
    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex((host, port)) == 0:
                print("Port is open")
                return True
            else:
                print("Port is open")
                return False
    except Exception as e:
        print(f"{e}")
        return False


def initialize_server_debugger():
    host = "0.0.0.0"
    port = 3000

    if getenv("DEBUG") == "True":
        import multiprocessing

        if multiprocessing.current_process().pid > 1:
            import debugpy

            try:
                debugpy.listen((host, port))
                print(
                    "⏳ VS Code debugger can now be attached, press F5 in VS Code ⏳",
                    flush=True,
                )
            except Exception as e:
                print(f"{e}")

            # debugpy.wait_for_client()
            # print("🎉 VS Code debugger attached, enjoy debugging 🎉", flush=True)
