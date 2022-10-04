import gdown
import os
import zipfile

os.mkdir('dataset')
os.chdir('dataset')
os.mkdir('images')
os.mkdir('labels')


os.chdir('images')
imgs = {
    '100K.zip': '1Ca_6X1c92gU0DzIORUZrqIL-PeJzA37C',
    '10k.zip': '1zTBEXeR4-RWo1L9716x9toBFfgw-a4vm',
}

for idx in imgs:
    gdown.download(id=imgs[idx], output=idx, quiet=False)
    with zipfile.ZipFile(idx, 'r') as zip_ref:
        zip_ref.extractall(idx.replace('.zip', ''))

    os.rmdir(idx)

os.chdir('../labels')
labels = {
    'bdd100k_det_20_labels_trainval.zip': '1cU97daBhJ2vMZLeavYY-f5-ZA0TqpxKD',
    'bdd100k_drivable_labels_trainval.zip': '18sxxr2gJ4aLb8Fut5kAViMtLRyksM9nT',
    'bdd100k_ins_seg_labels_trainval.zip': '1tPfG-9cXTK5j4zAYGMgTUzJGTGRmdDy9',
}

for idx in labels:
    gdown.download(id=labels[idx], output=idx, quiet=False)
    with zipfile.ZipFile(idx, 'r') as zip_ref:
        zip_ref.extractall(idx.replace('.zip', ''))

    os.rmdir(idx)
