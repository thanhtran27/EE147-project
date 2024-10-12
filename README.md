# Improving the Performance of KNN and Linear Regression on GPU Framework
Project Group - HIT <br />
Members:  Trong Tran, Isaac Castro, Howard Li <br />

## Overview
The main objective of this project is to implement algorithms such as K-Nearest
Neighbor (KNN) and Linear Regression on GPU in order to improve the performance by
leveraging GPU computing capabilities. By implementing and optimizing these algorithms for
GPU execution, we will compare their performance against traditional CPU implementations.
The goal is to speedup the computing process and gain efficiency, demonstrating the benefits of
GPU acceleration involving large datasets. <br />
The computation of KNN on CPU is done with python for a traditional implementation
compared to a CUDA implementation. For Linear Regression, we will compare the performance
of GPU and CPU with the batch version of Linear Regression for a stable and accurate update.

## How is the GPU used to accelerate the application?
The KNN algorithm’s distance computation step is parallelized on the GPU with each
thread responsible for calculating the distance between the data points. The overall space is
partitioned into threads in a 2D grid, with each thread responsible for calculating as mentioned
above. The kernel functions computation of distances(distance_compute), and the determination
of class labels(get_label) are parallelized stages involving heavy computation and are suitable for
parallel execution on the GPU.

### K-Nearest Neighbors
The KNN algorithm’s distance computation step is parallelized on the GPU with each
thread responsible for calculating the distance between the data points. The overall space is
partitioned into threads in a 2D grid, with each thread responsible for calculating as mentioned
above. The kernel functions computation of distances(distance_compute), and the determination
of class labels(get_label) are parallelized stages involving heavy computation and are suitable for
parallel execution on the GPU.

### Linear Regression
The Linear Regression algorithm’s gradient descent step is parallelized on the GPU. Each
thread calculates the gradient for a subset of data points and updates model parameters
concurrently. The problem space is partitioned into threads organized in a 1D grid. The
optimization process(gradient_descent_kernel) function is parallelized. This function involves
iterative computation of gradients and model parameter updates. Below are the linear regression
formula and the update rule for each regression weight.
![image](https://github.com/user-attachments/assets/3296022e-006c-49c6-a0a4-5f277a54f134)

## Implementation details
### K-Nearest Neighbors
The implementation of the KNN algorithm in CUDA mainly uses the GPU to launch two
kernel functions, ‘distance_compute’ and ‘get_label’.
1. Distance_compute kernel:
- Assign the thread position by using the function provided by Numba which is
cuda.grid(dim) where dim is the dimension. In this case, for computing the distance
matrix, dim is 2. <br />
```
i, j = cuda.grid(2)
```
- Allocate shared memory for a chunk of the dataset.<br />
```
shared_training = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE),dtype=float32)
```
- Use shared memory to load a chunk of training data to speed up computation by lowering
the time access to global memory. Also, use syncthread() to ensure all threads finish
loading before computing.<br />
```
if i < test_data.shape[0] and j < training_data.shape[0]:
for k in range(0, training_data.shape[1], 16):
idx = k + cuda.threadIdx.x
if idx < training_data.shape[1]:
shared_training[cuda.threadIdx.y, idx - k] = training_data[j, idx]
```
- Initialize a variable named squared_sum with value of 0.0 and use a for loop to iterate
through all the attributes of each training data to compute the squared sum.<br />
```
squared_sum = 0
for k in range(test_data.shape[1]):
diff = test_data[i, k] - shared_training[cuda.threadIdx.y, k]
squared_sum += diff * diff
```
- Store the result to global memory:
```
dis[i, j] = math.sqrt(squared_sum)
```
2. Get_label kernel:
- Assign the thread position by using the function provided by Numba<br />
```
i = cuda.grid(1)
```
- Allocate local memory for storing the nearest neighbor indices and the count for each
unique label found from a data point’s neighbors. Because we cannot dynamically modify
the size of the array, so we assume that the total of classes is 100, which is the<br />
MAX_LABELS<br />
```
if i < dis.shape[0]:
nearest_indices = cuda.local.array(shape=MAX_LABELS,
dtype=int32)<
label_counts = cuda.local.array(shape=MAX_LABELS,
dtype=int32)<
for k in range(MAX_LABELS):
label_counts[k] = 0
```
- Find the indices of k nearest neighbors by finding the smallest distance and the index of
that neighbor. Also, update the distance to be MAX_DISTANCE so that in the next
iteration, this neighbor is not considered again. <br />
```
for k in range(K):
min_index = -1 
min_distance = MAX_DISTANCE 
for j in range(dis.shape[1]): 
if dis[i, j] < min_distance: 
min_distance = dis[i, j] 
min_index = j 
nearest_indices[k] = min_index 
dis[i, min_index] = MAX_DISTANCE
```
- Each thread then counts the number of occurrences of labels among each data point’s
neighbors:
```
for k in range(K):
label = training_labels[nearest_indices[k]]
label_counts[label] += 1
```
- Check the local array of label_counts to find the highest number of occurrences of a label
and assign the global memory with that label.
```
max_label = -1
max_count = -1
for k in range(MAX_LABELS):
if label_counts[k] > max_count:
max_count = label_counts[k]
max_label = k
pred_labels[i] = max_label
```
### Linear Regression
The main kernel function implemented is the ‘gradient_descent_kernel’ function.
- Get the thread position by using the function provided by Numba. As the dimension of
our grid is one, so
```
idx = cuda.grid(1)
```
- Allocate shared memories to store a chunk of data and also the weight theta. In the
implementation, the block size is 128.
```
s_x = cuda.shared.array(shape=(BLOCK_SIZE, 2), dtype=float32)
s_y = cuda.shared.array(shape=BLOCK_SIZE, dtype=float32)
```
- Load a chunk of x with 2 features and y which is the independent variable into s_x and
the current parameter theta to s_theta. Use cuda.syncthreads() to ensure all threads finish
loading the data.
```
if idx < x.shape[0]:
s_y[cuda.threadIdx.x] = y[idx, 0]
for i in range(num_features):
s_x[cuda.threadIdx.x, i] = x[idx, i]
cuda.syncthreads()
```
- Each thread then computes the hypothesis function for each data point xi and compare
with its dependent variable yi of that data and update the corresponding parameter in
global memory by using atomic to prevent race conditions
```
diff = 0.0
if idx < x.shape[0]:
s_y[cuda.threadIdx.x] = y[idx, 0]
for i in range(num_features):
s_x[cuda.threadIdx.x, i] = x[idx, i]
cuda.syncthreads()
if idx < x.shape[0]:
hypothesis = 0.0
for i in range(num_features):
hypothesis += s_x[cuda.threadIdx.x, i] * theta[i]
diff = hypothesis - s_y[cuda.threadIdx.x]
for i in range(num_features):
cuda.atomic.add(theta, i, -learning_rate * diff *
s_x[idx, i])
```
## Documentation:
To run the code, follow these steps:
1. Open Google Colab
2. On the menu options, choose Runtime → Change runtime type → Choose T4 GPU
3. On the menu options, choose Edit → Open notebook. Then, drag the jpynb file in order
to load it in Google Lab.
4. On the left hand side, choose Folder and upload the Iris.data.
5. Choose Runtime → Run all to execute the code.

## Evaluation/Results:
### K-Nearest Neighbors:
We can see that the CPU runs roughly 22.4ms per loop whereas the GPU runs 2ms per
loop. This is roughly an 85% increase in speed on the GPU compared to the CPU. Moreover,
when comparing the computation time of the GPU with the well-known library scikit-learn, the
GPU still executes faster. Furthermore, the predicted labels produced by both the CPU and GPU
are identical, confirming that the GPU acceleration didn’t compromise the accuracy of the
program. However, the acceleration program can still be improved in the future since there’s still
under-utilization of the GPU likely due to low grid sizes. With more optimization the
performance could be enhanced even further.
![image](https://github.com/user-attachments/assets/86254fb2-47f6-4fe6-9ab7-73625754d2dd)
![image](https://github.com/user-attachments/assets/d4779423-f913-48b3-ac26-48cc0a957c8a)

### Linear Regression
For the Linear Regression acceleration program, the execution time of the GPU is
unexpectedly longer than the CPU’s time. As shown above, the CPU took roughly 38.4 ms
whereas the GPU took 65.7 ms to run per loop, indicating that the GPU didn’t not achieve the
anticipated speedup. However, both the CPU and GPU resulted in identical theta output,
indicating that the GPU is correctly implementing the Linear Regression process. The reason
why the GPU took longer to run is likely due to the kernel not being fully optimized for the
problem size, and atomic operations. Furthermore, the data set used(Iris) might not fully utilize
the GPU’s capabilities. Larger datasets might result in a better performance gain from the GPU.
![image](https://github.com/user-attachments/assets/bb2b21b2-3692-45cf-ae9b-2c50c5aa0a97)
![image](https://github.com/user-attachments/assets/5551e317-a4f8-48f5-8b33-bdd086d2e3b7)

## Problem Faced
The main challenge in this project is regarding the performance of executing Linear
Regression on the GPU framework. As mentioned above, the GPU implementation did not yield
the expected results. Further optimization can be implemented in regard to the small datasets and
the nature of the algorithm. Moreover, using shared memory and thread synchronization 
efficiently were another challenge as it could potentially bottleneck the overall performance. <br />
Experimenting with different grid and block sizes was conducted in order to find a balanced size
that maximizes GPU utilization without excessive overhead. In the K-nearest neighbor program,
the algorithm complexity was also another challenge as it’s inherently complex, but utilizing
parallel computation success distributed the workload among multiple threads allowing it to
perform efficiently and correctly.
