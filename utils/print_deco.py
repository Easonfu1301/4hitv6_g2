import time


Debug = False

CHECK_DEBUG = True


def print_decorator(func):
    def wrapper(*args, **kwargs):
        if Debug:
            func(*args, **kwargs)


    return wrapper


def only_check(func):
    def inner_func(*args, **kwargs):
        if CHECK_DEBUG:
            re = func(*args, **kwargs)
            return re



    return inner_func


def calculate_time(func):
    def inner_func(*args, **kwargs):
        start = time.time()
        re = func(*args, **kwargs)
        end = time.time()
        print(func.__name__,  " Time cost: ", end - start)
        return re

    return inner_func





# 应用装饰器到 print 函数
# print = print_decorator(print)