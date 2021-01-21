from socket import gethostname
from getpass import getuser


def get_host_info():
    return f'{getuser()}@{gethostname()}'
