---
title: "算法 笔试真题笔记"
---
# 数列

1. 根据值删除元素
(1) remove() 方法
删除列表中第一个匹配的元素，若元素不存在会抛出 ValueError。

```python
a = [1, 2, 3, 2, 4]
a.remove(2)  # 删除第一个值为2的元素
print(a)     # 输出: [1, 3, 2, 4]
```
(2) 循环删除所有匹配元素
若需删除所有匹配的值，需遍历列表的副本或反向遍历，避免索引错乱：

```
python
a = [1, 2, 3, 2, 4]
for num in a.copy():  # 遍历副本
    if num == 2:
        a.remove(num)
print(a)  # 输出: [1, 3, 4]

# 或反向遍历
a = [1, 2, 3, 2, 4]
for i in reversed(range(len(a))):
    if a[i] == 2:
        del a[i]
print(a)  # 输出: [1, 3, 4]
```

2. 根据索引删除元素
(1) pop() 方法
删除指定索引的元素并返回该元素。若不指定索引，默认删除最后一个元素。

```
python

a = [1, 2, 3, 4]
popped = a.pop(1)   # 删除索引1的元素（值为2）
print(a)            # 输出: [1, 3, 4]
print(popped)       # 输出: 2

a.pop()             # 删除最后一个元素（4）
print(a)            # 输出: [1, 3]
```
(2) del 语句
直接删除指定索引或切片范围的元素，无返回值。
```
python
a = [1, 2, 3, 4]
del a[1]        # 删除索引1的元素
print(a)        # 输出: [1, 3, 4]

del a[0:2]      # 删除索引0到1的元素
print(a)  
```
## 待debug例题
[开发商份土地](https://kamacoder.com/problempage.php?pid=1044)  

本来用二分法+两个前缀和，但是用以下代码可以发现：二分法没有直接用for循环节省内存。  
```python
try:
   代码
excep Exception as e:
  print(str(e)) 
```

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
- 指针逆向搜索，因为后面的字符不受前面输入影响
- 计数器统计#数量，当碰到非#字符，只有count=0，才能放入new list里，否则，只能让count数-1
- 字符串无法更改，只能新建一个list用来存放


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

**计算前缀和**
- 处理输入数据的时候，就可以开始计算前缀和了，无需储存原始数据
- 如果是0位，直接赋值，如果非0位，则用前一位的前缀和+当前输入
**利用前缀和计算区间和**
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

## 螺旋矩阵
[螺旋矩阵](https://leetcode.cn/problems/spiral-matrix-ii/)

纪念一下，第一次自己想的写的，内存耗时都不错
- 找准循环条件：螺旋的loop数，n//2. (奇数n在最后有一个元素单独填
- 更新各种状态：每个loop的起始位置start p, 起始数字：end num +1, 每个loop的边长
- 明确每层loop的四个顶点的坐标，坐标更新即某条轴坐标不动，另一条轴的坐标用 起始坐标+ 移动步数。而移动步数 = 当前数字i - 当前边长的初始顶点的数字
- 明确每层都是左闭右开原则，即起始顶点填，结束顶点不填，这样四条边走完，正好不覆盖

![效果](./pics/image.png)
```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        l = n//2 # 子正方形的个数
        is_odd = n % 2 # 最后是否有一个边长为1的正方形
        end_num = 0 # 实时更新每一个正方形结束后，填到哪个数字了
        start_p = 0 # 每个正方形的开始坐标（start_p, start_p） 和边长n 一起可确定四个顶点坐标
        mat = [[0]*n for _ in range(n)]
        for sl in range(l):
            for i in range(end_num + 1,end_num + n):
                mat[start_p][start_p+i-1-end_num] = i
            for i in range(end_num + n, end_num + 2*n - 1):
                mat[start_p+i-n-end_num][start_p+n-1] = i
            for i in range(end_num + 2*n - 1,end_num + 3*n -2):
                mat[start_p+n-1][start_p+n-1-(i-2*n-end_num+1)] = i
            for i in range(end_num + 3*n -2,end_num + 4*n -3):
                mat[start_p+n-1-(i-end_num-3*n+2)][start_p] = i
            end_num += 4*n -4
            n -= 2 #check n负数
            start_p += 1 
        if is_odd:
            mat[start_p][start_p] = end_num + 1
        return mat
```

# 链表
## 基本操作
- 记得加一个dummynode
- 链表循环的时候，判断好while的退出条件，即是要当前结点还是前一结点
- 判断好边界条件的处理：比如空链表和单节点链表

## [环形链表](https://leetcode.cn/problems/linked-list-cycle-ii/)

快慢指针法：
判断是否有环：利用环形性质，速度一快一满的两个指针，一定会再次重叠
判断含环链表的入口：根据快慢指针的速度特性，设计让两个指针能相遇在环状入口处

# 哈希表
set和dict都是无序的，list是有序的。
需要去重的时候，用set，除了元素本身，还要保存其他信息时可以用dict，当key值是连续的，比如全小写字符，就可以用list
**set**  
set是{},()是tuple，set内只能放入不可变对象，比如数值、字符串、none、元组（且元组内部同样是可哈希对象）。
可使用list来储存不重复的可变对象，每次加入前进行一次判断即可  
基础操作：
1.初始化
```python
s = {1, 2, 3}             # 直接定义
s = set([1, 2, 2, 3])     # 从列表去重创建 → {1, 2, 3}
s = set("hello")          # 从字符串创建 → {'h', 'e', 'l', 'o'}
```
2.添加
```python
s = {1, 2}
s.add(3)       # 添加单个元素 → {1, 2, 3}
s.update([4,5])# 批量添加 → {1, 2, 3, 4, 5}
```
3.删除
```python
s = {1, 2, 3}
s.remove(2)    # 删除元素（元素不存在则报 KeyError） → {1, 3}
s.discard(99)  # 删除元素（元素不存在不报错） → {1, 3}
val = s.pop()  # 随机删除并返回一个元素（集合为空时报错）
s.clear()      # 清空集合 → set()
```
4.集合
```python
# 并
a ｜ b
#交
a & b
# 差
a - b
# 子/超集合
a <= b, b >= a
```


**dict**
python 提供的两种特殊的字典
在 Python 的 collections 模块中，defaultdict 和 Counter 是两个常用的工具类，用于简化字典操作和统计任务。以下是它们的详细说明和区别：

1. defaultdict
作用
defaultdict 是字典 (dict) 的一个子类，在访问不存在的键时，自动为该键生成一个默认值，避免抛出 KeyError。你需要指定一个默认值的类型（工厂函数）。

使用场景
当需要按键分组或收集数据（例如将元素分组到列表中）。

需要避免频繁检查键是否存在。

```python
# 统计单词出现的位置（将位置索引存入列表）
words = ["apple", "banana", "apple", "cherry", "banana"]
word_positions = defaultdict(list)

for idx, word in enumerate(words):
    word_positions[word].append(idx)  # 直接追加，无需检查键是否存在

print(word_positions)
# 输出：defaultdict(<class 'list'>, {'apple': [0, 2], 'banana': [1, 4], 'cherry': [3]})
```

2. Counter
作用
Counter 是字典的子类，专门用于统计可哈希对象的出现次数。它简化了频率统计任务，提供常见统计方法（如获取最高频元素）。

使用场景
统计元素频率（如字符、单词、数字出现的次数）。

快速获取前 N 个高频元素。
```python
# 统计列表中元素的频率
data = ["a", "b", "a", "c", "b", "a"]
counter = Counter(data)

print(counter)
# 输出：Counter({'a': 3, 'b': 2, 'c': 1})

# 获取出现最多的前2个元素
print(counter.most_common(2))  # [('a', 3), ('b', 2)]
```

## [找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/)

连续子序列+滑动窗口：注意对每个子序列进行某些统计的时候，可以根据滑动情况，更新部分数据即可，不需要每次都对整个子序列进行计算

## [四数相加](https://leetcode.cn/problems/4sum-ii/solutions/499745/si-shu-xiang-jia-ii-by-leetcode-solution/)

答案解法：四个循环拆成两个两重循环，即两两数组相加
我的解法：一个数组一个数组逐个处理，每处理一个数组，就更新pre字典，pre字典key存放了前几个数组之和，val存放和等于该val的情况


## [四数之和](https://leetcode.cn/problems/4sum/submissions/610060030/)

答案解法：Counter +三重循环 ，在三重循环内部判断三数之和的complement是否在counter中
我的解法：三个字典分别存放，三数之和、两数之和，一数，主要是最外层一遍循环，每来了一个数字，按先后顺序，更新根据二数之和更新三数之和，根据一数更新二数

# 堆

1. heapq 模块
最小堆实现：heapq 是 Python 标准库中的模块，默认实现最小堆。 Python 的 heapq 模块中，当往最小堆中插入元组时，堆默认会按照元组的元素依次比较（即先比较第一个元素，若相同则继续比较第二个元素，依此类推）.最小堆就是一个二叉堆，即子结点的值都小于父结点

注意，最小堆的底层遍历默认是层次遍历，因此如果按照for循环去遍历，打印出来的很可能不是全部有序的，最小堆只保证父子之间的次序，不保证兄弟节点的次序。因此，要依次取出最小值，得用
```python
while l:
   heapq.heappop(l)
```

[1, 3, 2, 7, 4]，其堆结构为
```
      1
    /   \
   3     2
  / \
 7   4
```

核心函数：
初始化一个空堆: []即满足特性堆

heapify(iterable)：将列表原地转换为堆结构（时间复杂度 O(n)）。

heappush(heap, element)：插入元素并维护堆属性。

heappop(heap)：弹出堆顶（最小元素）。

heappushpop(heap, element)：先插入后弹出堆顶，比分开调用高效。

heapreplace(heap, element)：先弹出堆顶后插入新元素。

nlargest(n, iterable) / nsmallest(n, iterable)：获取前 n 个最大或最小元素（无需完全排序）。

2. 实现最大堆
数值取反法：通过插入元素的相反数模拟最大堆。

# 字符串
## ASIMA值
获取asima值：`ord()`  
判断小写字母、数字
```python
 ord("0") <= ord(s) <= ord("9")
 ord("a") <= ord(s) <= ord("z")
```

## 字符串基础操作及函数

split默认按照空格/tab等进行拆分，不管单词之间是多少个空格，都能拆分，包括开头的空格和末尾的也都可以删除  

' '.join():   字符串连接函数，以空格进行连接，要传递一个迭代器，

list.reverse(): 直接就地反转，返回list
reversed(list): 返回一个迭代器

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        word_list = s.split()
        return ' '.join(reversed(word_list))
```

取字符串后n位，取到倒数第n位，且切片能自动处理越界，比如字符串长度不足n位，则返回整个字符串

``` python
sr = s[-n:]
sl = s[0:-n]
```

find：获取字符串中子字符串的起始索引，如果没有则返回-1，子字符串可以是单个字符，也可以是多个
```python
s = "hello"
print(s.find("x"))  # 输出 -1
```

## 反转字符串二
收获：
1.模拟题
2. for循环，每次步长为2k，当超过数组范围，自动结束循环  
3. 数组切片的保护机制，当切片末尾超过索引范围，自动从末尾切断，实现不足k长度的，直接反转自身
```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        t = list(s)
        for i in range(0, len(t), 2 * k):
            t[i: i + k] = reversed(t[i: i + k])
        return "".join(t)
```

# 栈与队列
python中的双端队列
```python
from collections import deque
d = deque()
d.appendleft(1)   # 左侧入队 → [1]
d.append(2)       # 右侧入队 → [1, 2]
d.popleft()       # 左侧出队 → 1
d.pop()           # 右侧出队 → 2
```

## [滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

维护一个单调栈，栈内只保存可能成为最大值的元素，窗口滑动时
- 右侧元素入栈： 一直弹出左侧值，直到队尾值大于该元素
- 元素出栈：只有窗口外的这个值恰好等于栈头元素，才需要出栈。（因为窗口外的这个元素，如果不在队头，说明早就已经被淘汰掉，不可能成为最大值了，比如[1,2] 2 把1淘汰，栈内不会有1，根据元素入栈规则也可判断）

  
  
# 二叉树

## 层次遍历
有两种
1. 维护队列，队头弹出，并写入result中，队尾加入这个结点的左右子结点
2. 递归，思路和中序遍历一致，只不过每次加入的时候，将结点加入一个嵌套list所对应的层数, 递归函数需传递一个level参数
   

## [验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/description/)

给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。  

key: 
1. 递归思想，判断左边是否是一个BST，右边是否是一个BST，中间节点的值大于左边最大值，小于右边最小值
2. 关键在于节点间值大小的判断逻辑，要利用BST中序遍历的值递增性质，依次访问的节点的值一定是递增的，如果不满足，直接返回false即可
3. 那么，在每个递归函数内部，只需要做一个值判断，就是当前节点的值是否大于之前遇到过的最大值（也就是上一个节点的值）



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


# 回溯法
回溯法和二叉树的递归是一样的，只不过二叉树直接遍历左节点，再遍历右结点，而回溯法由于子节点数量不定，因此由for循环来代替逐个枚举  

区分回溯法和滑动窗口:
1. 滑动窗口本质是两个for循环，而回溯法则可以是任意个循环
2. 能使用回溯法的问题要满足，一个递归算法，内部一个for循环，即for循环下的每个子节点又是一个原问题（从二叉树角度理解）

回溯内进行递归时，往往会传入list来存放结果，并作为参数传递，要注意list此类可变对象，作为函数参数传递的时候，是传递的引用而非副本，因此需要注意两点
1. 递归函数无需显示返回，递归函数内部改变了list，外部也自动改变了
2. 需要永久保存该结果到另一个地方，并且避免之后该list被改动，一定要使用list副本，比如[:],切片自动生成副本

3. 
   


$$  
h(t)=\frac{e^t}{1+e^t}=\frac{1}{1+e^{-t}}  
$$   

![sigmod 函数图像](https://img2018.cnblogs.com/blog/790418/201811/790418-20181107181130984-1052306153.png)  



