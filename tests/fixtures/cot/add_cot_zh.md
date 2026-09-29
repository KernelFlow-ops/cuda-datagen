1. 题意与张量/入口
输入为长度 n 的 float 数组 `a` 与 `b`，输出 `c`，入口函数 `launch_vector_add` 接收设备指针与 n。

2. 算法
逐元素相加，每个输出元素只依赖同下标的两个输入，天然并行，无需归约。

3. 线程与 block 映射
一个线程负责一个元素，全局下标 i = blockIdx.x * blockDim.x + threadIdx.x；BLOCK = 256，grid 为 (n + BLOCK - 1) / BLOCK。

4. 存储与同步
相邻线程访问相邻地址，读写都是合并访存；线程间无数据依赖，不需要 __syncthreads。

5. 边界与异常
最后一个 block 可能越界，用 i < n 判断保护；n 为 0 时 grid 为 0，不启动 kernel 也安全。

6. 实现清单
kernel `vector_add_kernel` 完成计算，host 入口 `launch_vector_add` 计算 grid 并启动。
