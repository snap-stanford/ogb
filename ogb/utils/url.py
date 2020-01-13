import zipfile
import os
import os.path as osp
import errno
import requests

def decide_download(url):
    size_byte = int(requests.get(url).headers["content-length"])
    GBFACTOR = float(1 << 30)
    size = size_byte/GBFACTOR
    ### confirm if larger than 1GB
    if size > 1:
        return input("This will download %.2fGB. Will you proceed? (y/N)\n" % (size)).lower() == "y"
    else:
        return True

def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.
    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)
    tmp_download_path = osp.join(folder, "_" + filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print('Using exist file', filename)
        return path

    if log:
        print('Downloading', url)

    makedirs(folder)

    if osp.exists(tmp_download_path):
        resume_header = {'Range': 'bytes=%d-' % osp.getsize(tmp_download_path)}
        mode = "ab"
    else:
        resume_header = {}
        mode = "wb"

    with requests.get(url,
                      headers=resume_header,
                      stream=True,
                      verify=False,
                      allow_redirects=True) as r:
        with open(tmp_download_path, mode) as f:
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if chunk:
                    f.write(chunk)
    os.rename(tmp_download_path, path)
    return path

def maybe_log(path, log=True):
    if log:
        print('Extracting', path)

def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)

if __name__ == "__main__":
    url = "https://ogb.stanford.edu/data/pyg_mol_download/tox21.zip"
    ans = decide_download(url)
    print(ans)

