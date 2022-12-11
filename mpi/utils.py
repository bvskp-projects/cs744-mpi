import re


def camel_to_snake(name):
    """
    See https://stackoverflow.com/a/1176023/12160191
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
