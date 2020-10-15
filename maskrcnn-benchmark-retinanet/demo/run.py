from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
config_file = "/root/code/release/maskrcnn-benchmark-retinanet/configs/retina/retinanet_MobileNetV2-96-FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=324,
    confidence_threshold=0.5,
)
# load image and then run prediction
image = cv2.imread('/root/code/release/maskrcnn-benchmark-retinanet/demo/screen.jpg')
predictions = coco_demo.run_on_opencv_image(image)
cv2.imwrite("result.jpg", predictions)
