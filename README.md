# XAI-SCAN
This Project is working on eXplainable AI (XAI) for Clustering utilizing SCAN and Introspection approaches.<br>

## SCAN Clustering:
SCAN was imported from [here](https://github.com/wvangansbeke/Unsupervised-Classification)<br>
The SCAN README file can be found [here](SCAN_README.md)<br>

## XAI Methods:
### Introspection Implementation:
GradCam: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82 <br>
FeatViz: https://distill.pub/2017/feature-visualization / https://colab.research.google.com/github/tensorflow/lucid/blob/master/notebooks/feature-visualization/regularization.ipynb#scrollTo=gvF6j5h4GkDe <br>
<br>
### Cluster Word-Cloud:
Generate Image Descriptions with CLIP from OpenAI: https://huggingface.co/docs/transformers/model_doc/clip<br>
Find most frequent Terms and Generate Word-Clouds: https://towardsdatascience.com/generate-meaningful-word-clouds-in-python-5b85f5668eeb (maybe Projection on GradCam)
<br>

## Query Images:
```
> Exemplary Queries:

python singleInstance_varK.py -c 0,1,2 -k 5 -q ./data/cifar_img_5109.jpeg
python retrain_singleInstance_varK.py -c 50,250 -k 5 -e 50 -q ./data/cifar_img_5109.jpeg
```
For singleInstance_varK.py Script which does not train the networks again and hence is a quick query evaluation tool compared to the other script has the following command line argument setup: <br>
```
usage: singleInstance_varK.py [-h] [--config_pretext CONFIG_PRETEXT] 
                              [--config_main CONFIG_MAIN]
                              [--simclr_model SIMCLR_MODEL] 
                              [--scan_model SCAN_MODEL] [--save_path SAVE_PATH] 
                              [-k TOPK] [-c CLUSTER_HEADS] [-q PATHS_TO_IMG] 
                              [--re_calc] [--no_grad] 
                              [--no_viz] [--perf]

options:
  -h, --help            show this help message and exit
  --config_pretext CONFIG_PRETEXT
                        Location of simclr config file
  --config_main CONFIG_MAIN
                        Location of scan config file
  --simclr_model SIMCLR_MODEL
                        Location where model is saved
  --scan_model SCAN_MODEL
                        Location where model is saved
  --save_path SAVE_PATH
                        Location of save_paths
  -k TOPK, --topk TOPK  top k number for knn simclr method
  -c CLUSTER_HEADS, --cluster_heads CLUSTER_HEADS
                        comma separated index list of cluster heads 
                        (each head defines a number of k clusters) for scan
                        method (pretrained heads for k=[5,20,100,300,500])
  -q PATHS_TO_IMG, --paths_to_img PATHS_TO_IMG
                        paths to the sample img separated by comma no spaces
  --re_calc             re-calc knn for train set
  --no_grad             disable grad cam explainable ai functionality
  --no_viz              disable plot evaluation functionality
  --perf                used to time minimal query effort method
```
For retrain_singleInstance_varK.py Script which enables better specialisation and gives more adaptability   options the command line arguments look like this: <br>
```
usage: retrain_singleInstance_varK.py [-h] [--config_pretext CONFIG_PRETEXT] 
                                      [--config_main CONFIG_MAIN] 
                                      [--simclr_model SIMCLR_MODEL]
                                      [--simclr_checkpoint SIMCLR_CHECKPOINT] 
                                      [--scan_model SCAN_MODEL]
                                      [--scan_checkpoint SCAN_CHECKPOINT] 
                                      [--save_path SAVE_PATH] [-k TOPK]
                                      [-c CLUSTER_NUMBERS] 
                                      [--cluster_heads CLUSTER_HEADS] 
                                      [-q PATHS_TO_IMG] [-e EPOCHS] 
                                      [--no_grad] [--no_viz] [--perf]

options:
  -h, --help            show this help message and exit
  --config_pretext CONFIG_PRETEXT
                        Location of simclr config file
  --config_main CONFIG_MAIN
                        Location of scan config file
  --simclr_model SIMCLR_MODEL
                        Location where model is saved
  --simclr_checkpoint SIMCLR_CHECKPOINT
                        Location where model is saved
  --scan_model SCAN_MODEL
                        Location where model is saved
  --scan_checkpoint SCAN_CHECKPOINT
                        Location where model is saved
  --save_path SAVE_PATH
                        Location of save_paths
  -k TOPK, --topk TOPK  top k number for knn simclr method
  -c CLUSTER_NUMBERS, --cluster_numbers CLUSTER_NUMBERS
                        comma separated number of c clusters
  --cluster_heads CLUSTER_HEADS
                        comma separated index list of cluster heads 
                        (each head defines a number of k clusters) 
                        for scan method (pretrained heads for k=[5,20,100,300,500])
  -q PATHS_TO_IMG, --paths_to_img PATHS_TO_IMG
                        paths to the sample img separated by comma no spaces
  -e EPOCHS, --epochs EPOCHS
                        number of epochs for re-training phase
  --no_grad             disable grad cam explainable ai functionality
  --no_viz              disable plot evaluation functionality
  --perf                used to time minimal query effort method
```  
