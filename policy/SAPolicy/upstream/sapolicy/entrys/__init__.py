"""
Entry-point package.

Keep this module import-light: importing MuJoCo / robosuite / eval runners at
package import time can break training environments (and slows down CLI
startup). `main.py` dynamically imports the requested entrypoint module.
"""

__all__ = []
