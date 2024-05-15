#import zipfile module
from zipfile import ZipFile

with ZipFile('LDM_img.zip.zip', 'r') as f:

    f.extractall()