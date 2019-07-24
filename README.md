# what's this?
- Faster rcnn with run with least efforts, just follow the instructions :)
- Works with pytorch 1.0
- Models distributed

# Install
clone the repo.
```
git clone https://github.com/kentaroy47/over-the-counter-faster-rcnn.pytorch.git
```

install requirements.

```
pip install -r requirements.txt
```

Install the repo.
```
cd lib
python setup.py build develop
```

# Inference

This will do a res18 inference of your images.

The model is trained by MS-COCO.
```
python inference.py --image_dir images
```
