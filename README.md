# IMPROVING SAMPLE GENERATING AND SELECTION IN CONTRASTIVE LEARNING FOR PERSON RE-IDENTIFICATION 
The release code of the paper "Improving sample generating and selection in contrastive learning for person re-identification".  
  
## **Installation:**  
```
git clone https://github.com/andy412510/Contrastive-sample  
cd Contrastive-sample-main  
python setup.py develop  
```
The details of our development environment is decribed in:  
```
environment.txt  
```
## **Preparation:**  
- Download the datasets:  
Market-1501  
DukeMTMC-reID  

- Create original and 7 view person pose images. We use [HMR](https://github.com/akanazawa/hmr) in our work.  

- Put data under the directory:  

```
examples/data
├── market1501
│   └── bounding_box_train
│   └── bounding_box_test
│   └── ..
├── dukemtmc-reid
│   └── DukeMTMC-reID
│		└── bounding_box_train
│		└── ..

examples/mesh
├── market
│   └── train
│   	└── render
│   	└── render_45
│   	└── render_90
│   	└── ..
│   └── test
│   	└── render
│   	└── render_45
│   	└── render_90
│   	└── ..
├── DukeMTMC
│   └── train
│   	└── render
│   	└── render_45
│   	└── render_90
│   	└── ..
│   └── test
│   	└── render
│   	└── render_45
│   	└── render_90
│   	└── ..
```

## **Stage I: Warm up**  
Train an id encoder with unsupervised setting. We use [JVTC](https://github.com/ljn114514/JVTC) in our work.  


## **Stage II: GAN Training**  
Before training, make sure the stage is 2 and the path is correct in  
```
configs/config.yaml  
```
Make sure the setting is correct and run
```
examples/training.py  
```

## **Stage III: Final training**  
- GAN and contrastive model training

Make sure the stage is 3 and run
```
examples/training.py  
```

## **Results**  
- Experimental results  

Please see log files for more details in our work.  
```
logs/xxx.txt  
```

	