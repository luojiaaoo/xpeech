from .channel import restful
from asyncer import runnify


async def main():
    await restful.run()


if __name__ == '__main__':
    runnify(main)()
