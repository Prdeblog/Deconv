import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/DEConv-YOLO.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='data.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=4,
                close_mosaic=0,
                workers=4, 
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True,
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='pth',
                )