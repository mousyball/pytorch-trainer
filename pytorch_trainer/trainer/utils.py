from getpass import getuser
from socket import gethostname


def get_host_info():
    return f'{getuser()}@{gethostname()}'
