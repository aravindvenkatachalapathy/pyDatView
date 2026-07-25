__all__ = ['show', 'show_qt', 'show_sys_args']


# Keep GUI imports lazy so non-GUI tests and library imports do not load Qt.
def show(*args, **kwargs):
    from pydatview.qt_main import showApp
    showApp(*args, **kwargs)


def show_qt(*args, **kwargs):
    from pydatview.qt_main import showApp
    showApp(*args, **kwargs)

def show_sys_args():
    import sys
    if len(sys.argv) > 1:
        show(filenames=sys.argv[1:])
    else:
        show()
