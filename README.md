## Setup
### Install Package Dependencies
```
Python Environment: >= 3.6
torch >= 1.2.0
torchvision >= 0.4.0
numpy
```

### Install Datasets
We need users to declare a `base path` to store the dataset as well as the log of training procedure. The directory structure should be :


```
base_path
│       
└───dataset
│   │   OfficeHome
│       │  Art
│       │  Clipart
|       |  Product
│       │  RealWorld
│   │   DomainNet
│       │   ...
│   │   OfficeCaltech10
│       │   ...
|   |   Office31
|       |   ...

```
* DomainNet
  
  [VisDA2019](http://ai.bu.edu/M3SDA/) provides the DomainNet dataset.

* OfficeHome: 
Download zip file from [here](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view) (preprocessed by [SHOT](https://github.com/tim-learn/SHOT)) and unpack into `./data/OfficeHome65`. Verify the file structure to check the missing image path exist.
'''


## Run FDA experiments
```python
python main_train.py
```
Some code is borrowed from [KD3A](https://github.com/FengHZ/KD3A)