from datetime import datetime


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")
