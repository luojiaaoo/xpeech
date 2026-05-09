from pathlib import Path
a = Path('/c/b/a').relative_to(Path('/c')).as_posix()
print(a)