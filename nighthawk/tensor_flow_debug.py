"""Optional debug logging for model tensor / DataFrame inspection.

Enable with environment variable NIGHTHAWK_DEBUG=1.
"""

import os


def debug_enabled():
    return os.environ.get('NIGHTHAWK_DEBUG', '').lower() in ('1', 'true', 'yes')


def debug_section(title):
    if debug_enabled():
        print(f'\n=== {title} ===')


def debug_note(message):
    if debug_enabled():
        print(message)


def debug_tensor(name, value, note=None):
    if not debug_enabled():
        return

    print(f'[{name}] {type(value).__name__}', end='')
    shape = getattr(value, 'shape', None)
    if shape is not None:
        print(f' shape={tuple(shape)}', end='')
    print()
    if note:
        print(f'  {note}')
