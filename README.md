# Pytorch-Retinanet-real-time-vehicle-detection
Real time vehicle detection using MobilenetV2-Retinanet96. Trained on DETRAC 

![This is a alt text.](/result.jpg "This is a sample image.")

## INSTALL
- Follow instruction in install.md 
## Inference 
- Run demo/run.pu
## Benchmark 
- 6 FPS using Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz
|  STT  |         Model         | Confidence | DT_Box | AP(car) | MR(car) | AP(bus) | MR(bus) | AP(van) | MR(van) |  mAP   | mFPPI | Processing time |
| :---: | :-------------------: | :--------: | :----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :----: | :---: | :-------------: |
|   1   | Mb2-retina-fpn-96(v1) |    0.5     | 123515 | 64.99%  | 28.37%  | 69.98%  | 19.87%  | 37.20%  | 42.98%  | 47.33% |       |                 |

