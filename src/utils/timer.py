import time


def time_complexity(func):
    def wrapper(*args, **kwargs):
        if args[0].show_time_compute:
            start = time.time()
            results = func(*args, **kwargs)
            print('Time inference of retrieval: {}'.format(time.time() - start))
            return results
        else:
            return func(*args, **kwargs)
    return wrapper