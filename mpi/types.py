from dataclasses import dataclass

ClassType = str
Attrs = dict


@dataclass
class Arg:
    name: str
    attrs: Attrs


@dataclass
class Callable:
    args: list[Arg]
    return_attrs: Attrs


# TODO: Should we inherit from TypeError?
class CompileError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

class SynError(CompileError):
    pass

class SemError(CompileError):
    pass
