"""``python -m conninfpy`` — same as the ``conninfpy`` command."""
import sys

from conninfpy.cli import main

if __name__ == "__main__":
    sys.exit(main())
