## 将data传到GPU的过程与cpu异步

### Pinned Memory

默认情况下，host data的内存是可换页的。可换页内存的数据传到显存时，CUDA driver要先在host中开辟锁页的临时数组，把data传递到该数数组，再传到显存。

若可以直接将tensor存储于锁页内存，则可以减少开销。

!["data传输"](./Pinned%20Data%20Transfer.jpg "data传输")

### 
``` python
Tensor.cuda(device=None, non_blocking=False, memory_format=torch.preserve_format) → Tensor
```
该函数返回CUDA内存中关于该tensor的克隆。

要实现data异步传输到CUDA，可以设置non_blocking为True。注意，只有tensor存储于pinned memory上，该设置才能生效。设置后，复制过程相对于host将会是异步执行。






