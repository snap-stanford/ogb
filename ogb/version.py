import os
import logging
from threading import Thread

os.environ['OUTDATED_IGNORE'] = '1'
from outdated import check_outdated  # noqa


__version__ = '1.2.2'


def check():
    try:
        is_outdated, latest = check_outdated('ogb', __version__)
        if is_outdated:
            logging.warning(
                f'The OGB package is out of date. Your version is '
                f'{__version__}, while the latest version is {latest}.')
    except Exception:
        pass


thread = Thread(target=check)
thread.start()
