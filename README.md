# Pose Sensitive Embedding for Person Re-Identification (PSE)

In this repository, we provide the code used for our paper **A Pose-Sensitive Embedding for Person Re-Identification with Expanded Cross Neighborhood Re-Ranking**. 

This includes our training and prediction framework as well as the used neural network architectures and dataset readers. The Matlab code for our [Expanded Cross Neighborhood Re-Ranking](https://github.com/pse-ecn/expanded-cross-neighborhood) is located in a separate repository. This training framework is based on [Google's Tensorflow](https://www.tensorflow.org/) and the original network architectures (Resnet and Inception-v4) are inspired by the implementations provided in the [Tensorflow Models repository](https://github.com/tensorflow/models/tree/master/research/slim). All code is written in Python3 and was used with Tensorflow 1.3.

If you find our work helpful in your research, please cite:

```
M. Saquib Sarfraz, Arne Schumann, Andreas Eberle, Ranier Stiefelhagen,
"A Pose Sensitive Embedding for Person Re-Identification with Exapanded Cross Neighborhood Re-Ranking",
arxiv 2017
```


### Training a Model for Person Re-Identification
In order to train our models for Person Re-Identification, we start of with an Imagenet pre-trained model and fine tune it on our task. These pre-trained models can be found on the [Tensorflow Models Readme Page](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).

For training a Person Re-Id model, the `trainer_preid.py` script is used. To get a look at all possible arguments, have a look at the script's main method. Here, we'll only show the most important ones.

#### Preparing Datasets
To be able to use the dataset readers, simply extract the datasets downloaded from their project websites into an empty folder.

#### Supported Datasets
We currently support `duke`, `market1501` and `mars`. Although there are some other readers provided, we cannot give guarantees for them. To see the names used for training these datasets, have a look at the [DatasetFactory.py class](https://github.com/pse-ecn/pose-sensitive-embedding/blob/master/datasets/DatasetFactory.py).

#### Supported Network Architectures
To see the supported network's and their names, best have a look at the [nets_factory.py script](https://github.com/pse-ecn/pose-sensitive-embedding/blob/master/nets/nets_factory.py). Please note that the Tensorflow Models framework refers to the Resnet-50 network as `resnet_v1_50`. For the sake of compatibility, we did not change that.


#### Training a Resnet-50 Baseline on Market1501
To run our training script for a Resnet-50 Baseline model with market, execute the following while replacing all the <> tags with the corresponding values. 

`python3 trainer_preid.py --output=<output directory> --data=<dataset directory> --dataset-name=market1501 --batch-size=16 --num-epochs=100 --network-name=resnet_v1_50 --initial-checkpoint=<path to imagenet checkpoint or another checkpoint you want to load> --checkpoint-exclude-scopes=resnet_v1_50/logits --trainable-scopes=resnet_v1_50/logits --no-evaluation`

In this example, we use the optional parameters `--checkpoint-exclude-scopes` and `--trainable-scopes`. With the former, we can specify scopes to be excluded when loading the initial checkpoint (e.g. here, we exclude the logits, as they do not match between imagnet and Market1501). These layers will be randomly initialized. The latter one allows us to specify the scopes that should be trained. If this parameter is not specified, the whole network will be trained. 

For our work, we always started with an Imagenet pre-trained model and first trained only the randomly initialized layers before fine-tuning the whole network. To do this, you would start a new training after the one above finished or converged. For this new training, you specify that result as initial checkpoint and do not exclude scopes and train the whole network (not specify `--trainable-sopes`). 

To find out the scope names, please have a look at the network implementations in the `nets` package.


#### Evaluation during training
If you do not specify the `--no-evaluation` flag, the `trainer_preid.py` script will evaluate the model with the test and query set after every epoch. This is done by first predicting the test and query set and then using the Market1501 matlab evaluation script. Furthermore, the script will store the best checkpoint with the predicted features. Please note that this is not implemented for the MARS dataset as it takes a very long time predicting the features. These best features will be located in a subfolder of the `output` directory called `predictions-best`. 


#### Training the Views Predictor
In order to train our Views model, you need to have a dataset providing views information. In our paper, we used RAP for training the view predictor before transfering it to one of the other datasets. In contrast, our RAP dataset does not utilize person labels and thus can only be used for views training. The `trainer_views.py` script can be called as follows:

```
python3 trainer_views.py --output=<output directory> --data=<dataset directory> --dataset-name=market1501 --batch-size=16 --num-epochs=100 --network-name=resnet_v1_50_views --initial-checkpoint=<path to imagenet checkpoint or another checkpoint you want to load> --checkpoint-exclude-scopes=resnet_v1_50/logits --trainable-scopes=resnet_v1_50/3Views
```


#### Traiing Pose Map Models
The pose maps models are basically used the same way as the Baseline and the View models. However, to use them, you first need to generate the pose maps for the images. We did this by using the [Deeper Cut Tensorflow implementation](https://github.com/eldar/pose-tensorflow). For handling the pose maps, the datasets in our framework also have a `<dataset-name>-pose-maps` counterpart.

Therefore, to train a pose maps Resnet-50 model, you can run the following:

```
python3 trainer_preid.py --output=<output directory> --data=<dataset directory> --dataset-name=market1501-pose-maps --batch-size=16 --num-epochs=100 --network-name=resnet_v1_50 --initial-checkpoint=<path to imagenet checkpoint or another checkpoint you want to load> --checkpoint-exclude-scopes=resnet_v1_50/logits,resnet_v1_50/conv1 --trainable-scopes=resnet_v1_50/logits,resnet_v1_50/conv1 --no-evaluation
```

Please note that we mainly change the `--dataset-name` parameter to `market1501-pose-maps`. However, as the Imagenet pre-trained model expects only three input layers, we cannot use the first layer of that model. Therefore we not only exclude the *logits*, but also the *conv1* layer. Accordingly, we also add this layer to the trainable layers to enable the network to learn this layer in the first training step.


#### Training a Pose Sensitive Embedding Model (Views + Pose Maps Model)
Our PSE models are a combination of the Views Model and the Pose Maps Model. Therefore, for training the model, the training processes are also combined. The following command can be used to train a Resnet 50 PSE model.

```
python3 trainer_preid.py --output=<output directory> --data=<dataset directory> --dataset-name=market1501-pose-maps --batch-size=16 --num-epochs=100 --network-name=resnet_v1_50_views --initial-checkpoint=<path to model with RAP pre-trained views predictor> --checkpoint-exclude-scopes=resnet_v1_50/logits --trainable-scopes=resnet_v1_50/logits,resnet_v1_50/pre_logits,resnet_v1_50/3ViewBranches --no-evaluation
```

Herem we use the dataset version providing pose maps (`market1501-pose-maps`) with the network architecture utilizing the views prediction (`resnet_v1_50_views`). It is important to note that we have to use an initial model with a trained views predictor. This is the case because we do not have view labels for the Market1501 and Duke dataset. Again, we first train all randomly initialized layers before fine-tuning the whole model afterwards.



#### Using Tensorboard to track training
During training, you can keep track of the loss and other important numbers by starting Tensorboard.

```tensorboard --logdir=<output directory of the training or a parent folder of it>```

The results can be viewed by opening a browser and go to `localhost:6006`.


### Feature Prediction
To predict the features, run the `predictor_preid.py` script:

```python3 predictor_preid.py --model-dir=<the model to be loaded> --data=<dataset directory> --dataset-name=market1501 --batch-size=128 --network-name=resnet_v1_50_views```

The predicted features will be stored in a subfolder of the specified `model-dir` called `predictions`.



### General Notes

If you have any trouble using our code or find a bug or a mistake in this manual, please file a Github issue.