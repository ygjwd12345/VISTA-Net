"""Prepare PASCAL Context dataset"""
import os
import shutil
import argparse
import tarfile
from encoding.utils import download, mkdir

_TARGET_DIR = os.path.expanduser('./pcontext')
PASD_URL="https://codalabuser.blob.core.windows.net/public/%s"

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize PASCAL Context dataset.',
        epilog='Example: python prepare_pcontext.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args

def download_ade(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
         'bf9985e9f2b064752bf6bd654d89f017c76c395a'),
        ('https://codalabuser.blob.core.windows.net/public/trainval_merged.json',
         '169325d9f7e9047537fedca7b04de4dddf10b881'),
        ('https://hangzh.s3.amazonaws.com/encoding/data/pcontext/train.pth',
         '4bfb49e8c1cefe352df876c9b5434e655c9c1d07'),
        ('https://hangzh.s3.amazonaws.com/encoding/data/pcontext/val.pth',
         'ebedc94247ec616c57b9a2df15091784826a7b0c'),
        ]
    download_dir = os.path.join(path, 'downloads')
    mkdir(download_dir)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        if os.path.splitext(filename)[1] == '.tar':
            with tarfile.open(filename) as tar:
                tar.extractall(path=path)
        else:
            shutil.move(filename, os.path.join(path, 'VOCdevkit/VOC2010/'+os.path.basename(filename)))

def install_pcontext_api():
    repo_url = "https://github.com/zhanghang1989/detail-api"
    os.system("git clone " + repo_url)
    os.system("cd detail-api/PythonAPI/ && python setup.py install")
    shutil.rmtree('detail-api')
    try:
        import detail
    except Exception:
        print("Installing PASCAL Context API failed, please install it manually %s"%(repo_url))


if __name__ == '__main__':

    download_ade(_TARGET_DIR, overwrite=False)
    install_pcontext_api()