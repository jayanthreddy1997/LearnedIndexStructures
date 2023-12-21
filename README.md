# LearnedIndexStructures

## Abstract
Traditional indexing methods, while efficient, are not adaptive to varying data distributions. This limits their
performance in diverse, real-world applications. Neural networks are good at learning unknown data distributions.
We use this property of neural networks to build Learned Indexes tuned to the specific datasets. 
This theoretically would allow us to perform index lookups in O(1) amortized time as compared to traditional indexes
like B-Trees which take O(log(n)) time. 

![](https://github.com/jayanthreddy1997/LearnedIndexStructures/blob/main/static/learned_index.png)

## Project Details
We experimented with different model architectures and decided to use a 2 layer neural network with 128 neurons
in each layer. We used the ReLU activation function and ADAM optimizer. The model complexity was balanced to ensure
good predictive performance while limiting complexity to ensure low inference latencies. We also used GPUs for inference
which resulted in a significant speedup over benchmark results. 
Model performance was further improved by JIT-compiling PyTorch code into optimized kernels.

![Neural network architecture](https://github.com/jayanthreddy1997/LearnedIndexStructures/blob/main/static/model_architecture.png)

We evaluated the model's performance over three datasets with different distributions:
- normal_200M_uint32: 200 million unsigned 32-bit integers generated from a normal (Gaussian) distribution
- lognormal_200M_uint32: 200 million unsigned 32-bit integers drawn from a lognormal distribution
- uniform_sparse_200M_uint32: 200 million unsigned 32-bit integers that are sparce and uniformly distributed


## Results
All experiments were ran on a 24-core Intel Xeon Platinum 8268 processor with 32GB V100 Nvidia GPU and 12.2 CUDA version.  
- Since our goal is to learn the training data distribution, we could overfit on the training data. 
- Loss metric used was Mean Square Error.
- All datasets converged to < 8E-7 loss in 100 epochs taking less than 60s to train.
- Peak memory consumption ~100M (an unclustered B-Tree index takes > 1GB)
- R-squared nearly 1
- Query latency of around 1ms for 100k queries, which translates to 4.5x speedup over benchmark implementation (SOSD RMI). 
Speedup is around 40x over benchmark BTree implementation.

![Model performance](https://github.com/jayanthreddy1997/LearnedIndexStructures/blob/main/static/model_perf.png)

![Query Times](https://github.com/jayanthreddy1997/LearnedIndexStructures/blob/main/static/query_perf.png)

![Speedups](https://github.com/jayanthreddy1997/LearnedIndexStructures/blob/main/static/speedups.png)

## Conclusions
- Our Model’s query time due to its optimizations is significantly faster than traditional indexing methods 
- Primary focused was to improve read performance, although we have ensured the model can retrain in less than a minute when it receives writes.
- Model performs well with diverse data distributions, this demonstrates its robustness.
- Memory consumption is <10% of what a B-Tree would consume


## Project Milestones
- [x] Literature review
- [x] Train and Experiment with different Neural Network architectures
- [x] Optimize inference performance
- [x] Testing

## Description of the repository and code structure
- Configuration for each dataset and environment setting can be found in the YAML files in the `config` folder.  
- `dataloader.py` contains the dataloaders  
- `model.py` contains the main neural network model  
- `learned_index.py` contains the LearnedIndex class which uses the learned weights to predict the location of a key in a sorted array  
- `utils.py` contains few utility functions
- `tests.py` and `inference_time_tests.py` contain test methods for inference time, model performance, memory performance, etc


## Commands to run experiments
- Script to download datasets: https://github.com/learnedsystems/SOSD/blob/master/scripts/download.sh
- Create runtime: `conda create --name <env_name> --file requirements.txt`
- Run training for all datasets: `python main.py`
- Run inference performance tests: `python inference_time_tests.py`
- Run R-squared and memory stats tests: `python tests.py`


