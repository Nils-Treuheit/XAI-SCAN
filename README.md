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
