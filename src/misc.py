import psutil

def free_ram():
    return psutil.virtual_memory().available