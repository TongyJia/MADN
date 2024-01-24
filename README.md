# MADN
https://github.com/dehazing/MADN
## Effective Meta-Attention Dehazing Networks for Vision-Based Outdoor Industrial Systems 
![image](https://github.com/TongyJia/MADN/blob/main/dehazed_net.jpg)
### Prerequisites 
   　 Python3.6
     
   　 PyTorch>=1.0.1
     
   　 torchvision>=0.4.2
     
   　 skimage
     
   　 tqdm
   
   
### Quick Start
#### Testing:
Clone this repo in environment that satisfies the prerequisites

Run test.py using default hyper-parameter settings.

(You can test on the trained model:
 dehaze_80.pth is the pre-trained meta network,
 model.pth is the pre-trained dehazing network.)
 
#### Training:
 
 Run main.py for training.
 
 ### Cite
 If you use part of this code, please kindly cite
 
 ```
 @article{jia2021effective,
  title={Effective Meta-Attention Dehazing Networks for Vision-Based Outdoor Industrial Systems},
  author={Jia, Tongyao and Li, Jiafeng and Zhuo, Li and Li, Guoqiang},
  journal={IEEE Transactions on Industrial Informatics},
  year={2021},
  publisher={IEEE}
}
```
 
  



