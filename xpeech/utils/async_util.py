from asyncer import asyncify
import inspect

def ensure_async(func):
    if inspect.iscoroutinefunction(func):
        return func
    else:
        return asyncify(func)