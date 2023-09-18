import time


def timer(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print("{}: {}".format(func.__name__, end - start))

    return result
