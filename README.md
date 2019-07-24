# what's this?
- Faster rcnn with run with least efforts, just follow the instructions :)
- Works with pytorch 1.0
- Models distributed

# Install
install requirements.

```
pip install -r requirements.txt
```

Download pretrained coco model

```
wget https://www.dropbox.com/s/k68qq6wupseci7t/faster_rcnn_500_40_625.pth
```
cd lib
python setup.py build develop
```

# Inference

This will do a res18 inference of your images.

The model is trained by MS-COCO.
```
python inference.py --image_dir path-to-your-image-dir
```
