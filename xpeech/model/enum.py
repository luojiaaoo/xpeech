from enum import StrEnum


class ChannelType(StrEnum):
    WECHAT = "wechat"

class FormType(StrEnum):
    SINGLE = "single"
    MULTIPLE = "multiple"
    INPUT = "input"
