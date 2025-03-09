---
title: "阿里算法 笔试真题笔记"
---
# 数列

## 有序数组查找: 二分法
- 二分法的核心：用中间点的数据进行判断，决定是留下左半边区间，还是右半边区间，或者直接返回中间点
- 注意：
- 决定二分法的更新区间是 **左闭有闭** or **左闭右开** 是 `while right-left >= 0` or `while right-left > 0`
- 统一用左闭右闭，右端点更新时：middle-1， 左端点更新时：left = middle+1
-  middle = left + (right-left)//2
- 注意while结束后仍然没找到，return -1

## 移除元素
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素。元素的顺序可能发生改变。然后返回 nums 中与 val 不同的元素的数量。
### 我的解法：
计数器 count 统计已经碰到的要删除元素的个数，也即之后的元素要前移的长度  

for循环遍历，每个元素i前移count

### 标准解法
快慢指针法：fast指向正在处理的待移动元素，slow指针指向待被移入的位置  
首尾指针法

while fast < size

## 例题：[比较含退格字符串](https://leetcode.cn/problems/backspace-string-compare/submissions/607868111)
- 计数器统计#，
- 字符串无法更改，只能新建一个list用来存放
- 指针搜索，逆向

## [长度最小字串](https://leetcode.cn/problems/minimum-size-subarray-sum/submissions/607924611)
给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其总和大于等于 target 的长度最小的 子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
- 滑动窗口
- 思想：两个指针分别指着窗口的头和尾，这个窗口不断向后滑动。
- 1. 初始head不动，tail不断向后，直至找到满足条件的。当前长度的字串已经找到，其他同样长度的字串无需再找
  2. head后移一位，维持当前长度，继续不断向后移动。
  3. 如果又找到满足条件的。重复第二步，如果tail到达尾部，结束循环
     注意：边界情况在第一步出现，head不动，tail不断向后，可能出现整个数组也不满足条件
     
 ```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        ind1 = 0
        ind2 = 1
        min_l = []
        while ind1 <= len(nums)-1:  # 循环体条件： 
            if sum(nums[ind1:ind2]) >= target:
                min_l = nums[ind1:ind2]
                ind1 += 1
                
            else:
                if ind2 == len(nums) and len(min_l) == 0:
                    return 0
                if ind2 < len(nums):
                    ind2 += 1
                if len(min_l) > 0: 
                    ind1 += 1
                
        return len(min_l)
```

Note: 以上代码会有runtime错误，因为sum操作本质也是一层循环遍历，因此要将求和放在while内实现  

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        ind1 = 0
        ind2 = 0
        cur_l = len(nums) + 1
        cur_s = nums[ind1]
        while ind1 <= len(nums)-1:   # 循环体退出条件：在我的指针下，ind1和ind2指同一个位置是有意义的，代表长度为1，因此ind1可以指到最后一位len(nums)-1
            if cur_s >= target:
                cur_l = ind2 - ind1 + 1 
                cur_s -= nums[ind1]   
                ind1 += 1   # 这边加超之后，循环体条件破坏，就退出，上一行索引不会超
            else:
                if ind2 == len(nums)-1 and cur_l > len(nums):
                    return 0  # 这个if语句，是ind1可能一直在0位，while循环体无法退出
                if ind2 < len(nums)-1:
                    ind2 += 1  
                    cur_s += nums[ind2]      # cur_s 的累加，head指针ind2 先加，再累和
                if cur_l < len(nums) + 1: 
                    cur_s -= nums[ind1]    # cur_s 先减，tail指针再前移
                    ind1 += 1  
                      
        return cur_l if cur_l < len(nums) + 1 else 0
```

## 数组前缀和
如果查询m次，每次查询的范围都是从0 到 n - 1, 那么该算法的时间复杂度是 O(n * m).  前缀和的思想是重复利用计算过的子数组之和，从而降低区间查询需要累加计算的次数。 

**note:**
- 使用前缀和计算区间[a,b]的和时，用arr\[b] - arr\[a-1]
- 但如果a=0的话，直接就是arr\[b]


```python
import sys
l = int(input())
# arr = []
pre_sum = []
for _ in range(l):
    a = int(input())
    # arr.append()
    if len(pre_sum)==0:
        pre_sum.append(a)
    else:
        pre_sum.append(a+pre_sum[-1])


for line in sys.stdin:
    a,b = map(int,line.split())
    if a == 0:
        print(pre_sum[b])
    else:
        print(pre_sum[b]-pre_sum[a-1])
```


# 奇怪二叉树

> 二叉树 位运算 LCA
## 题目描述
[奇怪二叉树](https://www.neituiya.com/oj/7/725)

## 思路  

1.首先得到正常的完全二叉树编号 ： 正常的完全二叉树方便进行各种操作  

2. 对于两个结点，首先将深层结点（l1）向上移动到浅层结点的同一层l2
 - 如果这时两个结点一致，则路径极为层差l1-l2
 - 如果不一致，找这两个结点的公共母结点所在层l3，则路径为(l3-l2)*2+l1-l2

## 收获

完全二叉树标准化操作：  

- 将结点x上移一层：import math x>>1 或 x // 2
- 将结点x上移d层：x>>d
- 查询x所在层：一直将结点x上移，x>>=1, 直至x=0
- 查询同层节点的LCA：对这两个结点循环执行上移，直至相遇，循环中计数

读取输入

```python
q = int(input())
for _ in range(q):
    x, y = map(int, input().split())
```  


   


$$  
h(t)=\frac{e^t}{1+e^t}=\frac{1}{1+e^{-t}}  
$$   

![sigmod 函数图像](https://img2018.cnblogs.com/blog/790418/201811/790418-20181107181130984-1052306153.png)  



