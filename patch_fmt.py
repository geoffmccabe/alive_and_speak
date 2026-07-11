"""
Patches /app/models/float/FMT.py to ensure self.opt = opt is set
immediately after super().__init__() inside FlowMatchingTransformer.__init__
before any self.opt usage on line 203.
"""
import sys

path = "/app/models/float/FMT.py"

with open(path, "r") as f:
    lines = f.readlines()

# Find the FlowMatchingTransformer __init__ and inject self.opt = opt
# right after the super().__init__() call
in_fmt_class   = False
in_fmt_init    = False
injected       = False

new_lines = []
for i, line in enumerate(lines):
    # Detect we're inside FlowMatchingTransformer class
    if "class FlowMatchingTransformer" in line:
        in_fmt_class = True

    # Detect we're inside __init__
    if in_fmt_class and "def __init__(self, opt)" in line:
        in_fmt_init = True

    # After super().__init__() in the right __init__, inject self.opt = opt
    if in_fmt_init and not injected and "super().__init__()" in line:
        new_lines.append(line)
        # Detect indentation from the super() line
        indent = len(line) - len(line.lstrip())
        new_lines.append(" " * indent + "self.opt = opt  # patched by patch_fmt.py\n")
        injected = True
        continue

    new_lines.append(line)

if injected:
    with open(path, "w") as f:
        f.writelines(new_lines)
    print("SUCCESS: FMT.py patched — self.opt = opt injected after super().__init__()")
else:
    # Check if it's already there correctly
    content = "".join(lines)
    if "self.opt = opt" in content:
        print("INFO: self.opt = opt already present in FMT.py — no patch needed")
        sys.exit(0)
    else:
        print("ERROR: Could not find injection point in FMT.py")
        print("Lines around FlowMatchingTransformer.__init__:")
        for i, l in enumerate(lines):
            if "FlowMatchingTransformer" in l or "def __init__" in l or "super()" in l:
                print("  %d: %s" % (i+1, l.rstrip()))
        sys.exit(1)
