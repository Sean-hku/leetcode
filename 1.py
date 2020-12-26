# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#
# class Solution:
#     def maxDepth(self, root: TreeNode) -> int:
#         if root is None:
#             return 0
#         stack = [(1, root)]
#         depth = 0
#         while stack:
#             cur_dep, node = stack.pop()
#             depth = max(depth, cur_dep)
#             if node.right:
#                 stack.append((cur_dep+1,node.right))
#             if node.left:
#                 stack.append((cur_dep+1,node.left))
#         return depth
# root=TreeNode([3,9,20,None,None,15,7])
# a=Solution()
# # root=[3,9,20,None,None,15,7]
# print(a.maxDepth(root))
# N=2
# dp = [0] * (N+1)
# print(dp)
'''
def isValid(s: str) -> bool:
    dic = {')': '(', ']': '[', '}': '{'}
    stack = []
    for i in s:
stack.if
if stack and i in dic:
            if stack[-1] == dic[i]:
                stack.pop()
            else:
                return False
        else:
            stack.append(i)

    return not stack

print(isValid('(){}[]'))
'''
# a=[1,2,3,41,1,12,2]
# a.sort()
# print(a)
'''
def findUnsortedSubarray( nums) -> int:
    temp = nums[:]
    nums.sort()
    n = len(nums)
    for i in range(n):
        if temp[i] == nums[i]:
            i += 1
        else:
            break
    for j in range(n):
        if temp[-(j + 1)] == nums[-(j + 1)]:
            j += 1
        else:
            break
    return (n - i - j)
print(findUnsortedSubarray([2,6,4,8,10,9,15]))
'''


# def lengthOfLongestSubstring(s) -> int:
#     if not s:
#         return 0
#     length = []
#     maxlength = 0
#     for i in s:
#         if i not in length:
#             length.append(i)
#         else:
#
#             length[:] = length[length.index(i) + 1:]
#             length.append(i)
#         maxlength = max(maxlength, len(length))
#     return maxlength
#

# dic = {1: 2, 3: 4 }
# a=1
# def isValid(s) -> bool:
#     dic = {')': '(', ']': '[', '{': '}'}
#     stack = []
#     for i in s:
#         if  i in dic:
#             if stack in dict:
#                 if stack[-1] == dic[i]:
#                     stack.pop()
#                 else:
#                     return False
#         else:
#             stack.append(i)
#
#     return not stack
# isValid('{}(){}')
# b=2
# if b and a in dic:
#     print(111)

# def findUnsortedSubarray( nums) -> int:
    # bb = nums.sort()
#     bb = sorted(nums)
#     maxi = 0
#     mini = 0
#     flag = 1
#     for i in range(len(nums)):
#         if nums[i] != bb[i] and flag:
#             mini = i
#             flag -= 1
#         if nums[i] != bb[i]:
#             maxi = i
#     return (maxi - mini + 1)
# print(findUnsortedSubarray([2,6,4,8,10,9,15]))
# def combinationSum2( candidates, target) :
#     n = len(candidates)
#     result = []
#     candidates.sort()
#     def back(idx,tmp_num,tmp_list):
#         if tmp_num == target :
#             result.append(tmp_list)
#             return
#         for i in range(idx,n):
#             if tmp_num+candidates[i]>target:
#                 break
#             back(i+1,tmp_num+candidates[i],tmp_list+[candidates[i]])
#     back(0,0,[])
#     return result
# print(combinationSum2([10,1,2,7,6,1,5],8))
# import sys
# sys.setrecursionlimit(100000)
# def coinChange(coins, amount) :
#     n = len(coins)
#     result = []
#     coins.sort(reverse=True)
#     aa = 0
#
#     def back(idx, tmp_num, tmp_list):
#         print(result)
#         if tmp_num == amount:
#             result.append(tmp_list)
#             return
#         for i in range(idx, n):
#             # print(1)
#             if tmp_num > amount:
#                 break
#             back(i, tmp_num + coins[i], tmp_list + [coins[i]])
#
#     back(0, 0, [])
#     if not result:
#         return -1
#     return min([len(i) for i in result])
# def coinChange( coins, amount: int) -> int:
#     dp = [float("inf")] * (amount + 1)
#     dp[0] = 0
#     for i in range(1, amount + 1):
#         for coin in coins:
#             if (i >= coin):
#                 dp[i] = min(dp[i], dp[i - coin] + 1)
#     return dp[-1] if (dp[-1] != float("inf")) else -1
#
#
# print(coinChange([3,7,4,3],67))
# def maxSubArray( nums) -> int:
#     size = len(nums)
#
#     if size == 0:
#         return 0
#     dp = [0 for _ in range(size)]
#
#     dp[0] = nums[0]
#     for i in range(1, size):
#         dp[i] = max(dp[i - 1] + nums[i], nums[i])
#     print(dp)
#     return max(dp)
# def lengthOfLIS( nums) -> int:
#     n = len(nums)
#     dp = [1] * n
#     for i in range(1, n):
#         for j in range(i):
#             if nums[j] < nums[i]:
#                 dp[i] = max(dp[i], dp[j] + 1)
#                 print(dp)
#     return max(dp or [0])
#
# print(lengthOfLIS([-2,1,-3,4,-1,2,1,-5,1]))
# def reverseLeftWords( s: str, n: int) -> str:
#     s = list(s)
#     for i in range(n):
#         s.append(s[i])
#     return s[n:]
#
#
# print(reverseLeftWords('abcdefg',5))

# def findContinuousSequence(target):
#     aa = [i for i in range(1,int(target / 2) + 2)]
#     bb = []
#     for i in range(int(target / 2) ):
#         result = 0
#         j = i
#         while (result < target):
#             # print(i)
#             result += aa[i]
#             i += 1
#             if result == target:
#                 bb.append(aa[j:i])
#
#     return bb
import collections
#
# dic = collections.Counter([1,1,1,1,1,1,2,3,4,5,6,3])
# # print(dic)
# def twoSum( nums, target):
#     adic = {}
#     for i in nums:
#         if i in adic:
#             adic[i] += 1
#         else:
#             adic[i] = 1
#     for i in nums:
#         if (target - i) in adic and target != i * 2:
#             return [i, target - i]
# print(twoSum([14,15,16,22,53,60]
# ,76))
#
# def exchange( nums) :
#     a = 0
#     for i in range(1,len(nums)):
#         if nums[i] % 2 == 0:
#             a+=1
#             nums.append(nums[i])
#             # nums.remove(nums[i])
#     return nums[a:]
# print(exchange([2,16,3,5,13,1,16,1,12,18,11,8,11,11,5,1]))
# a=[1,2,3]
# a[0],a[2] = a[2],a[0]
# # a.pop(1)
# # print(list(range(5)))
# while True:
#     for i in range(5):
#         a=1
#     if a ==1:
#         break
#     print(11)
# print(22)
# import sys
# K, N = map(int,sys.stdin.readline().strip().split())
# Num = list(map(int, input().strip().split()))
# # print(Num)
# result = 0
# count=0
# flag = 1
# f =1
# flag2=1
# flag3=1
# if K<Num[0]:
#     for i in range(N):
#         if  result < K and flag == 1:
#             result = result + Num[i]
#             abs(result)
#         else:
#             result = result - Num[i]
#             # abs(result)
#         if result > K or abs(result) != result:
#             count += 1
#             result = abs(result)
#             result = result % K
#             flag -= 1
#         if result == K:
#             print(i)
#             f -= 1
#             print('paradox')
#     if sum(Num) < K:
#         print(K - result, count)
#         flag2 -= 1
#     if f and flag2 == 1:
#         print(result, count)


# for i in range(N):
#     if count==0 or result<K and flag==1 :
#         result= result+Num[i]
#         abs(result)
#     else:
#         result=result-Num[i]
#         # abs(result)
#     if result>K or abs(result)!=result :
#         count+=1
#         result=abs(result)
#         result= result%K
#         flag-=1
#     if result==K:
#         print(i)
#         f-=1
#         print('paradox')
# if sum(Num)<K:
#     print(K-result, count)
#     flag2-=1
# if f and flag2==1:
#     print(result,count)



# print(abs(-1))






# print(n,m)
# import sys
# land = []
# for i in range(6):
#     line = sys.stdin.readline().strip()
#     values = list( line.split())
#     land.append(values)
# print(land)
# def findNumberIn2DArray( matrix, target: int) -> bool:
#     m, n = len(matrix) - 1, 0
#     # for i in range(m): 不知道循环次数 用while
#     while m >=0 and n < len(matrix[0]):
#         print(m,n)
#         if matrix[m][n] > target:
#             m -= 1
#         elif matrix[m][n] < target:
#             n += 1
#         else:
#             return True
#     return False
# print(findNumberIn2DArray([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
# ,20))

# def maxValue( grid) -> int:
#     m, n = len(grid[0]), len(grid)
#     dp = [[0] * m for i in range(n)]
#     for i in range(m):
#         for j in range(n):
#             if i == 0 and j == 0:
#                 dp[j][i] = grid[0][0]
#             elif i == 0 and j != 0:
#                 dp[j][i] = dp[j - 1][i] + dp[j][i]
#             elif i != 0 and j == 0:
#                 dp[j][i] = dp[j][i - 1] + dp[j][i]
#             else:
#                 dp[j][i] = max(dp[j - 1][i], dp[j][i - 1]) + dp[j][i]
#     return dp[-1][-1]

# def maxProfit( prices) -> int:
#     minprice = float('+inf')
#     maxprofit = 0
#     for i in range(len(prices)):
#         minprice = min(minprice, prices[i])
#         maxprofit = max(prices[i] - minprice, maxprofit)
#     return maxprofit
# print(maxProfit([7,1,5,3,6,4]))

# a= list(range(5))
# print(a[1:3])

# Num = list(map(int, input().strip().split()))
# # print(Num)
# res=[]
# while True:
#     try:
#         s =input()
#         res.append(list(map(int,s.split(' '))))
#     except:
#         break
# def calc_area(rect1, rect2):
#      xl1, yb1, xr1, yt1 = rect1
#      xl2, yb2, xr2, yt2 = rect2
#      xmin = max(xl1, xl2)
#      ymin = max(yb1, yb2)
#      xmax = min(xr1, xr2)
#      ymax = min(yt1, yt2)
#      width = xmax - xmin
#      height = ymax - ymin
#      if width <= 0 or height <= 0:
#          return 0
#      cross_square = width * height
#      return cross_square
# def cal_area(rec):
#     x1,y1,x2,y2=rec
#     area=abs(x2-x1)*abs(y2-y1)
#     return area
# for i in res:
#     rect1 =i[0:4]
#     rect2 = i[4:8]
#     cross=calc_area(rect1,rect2)
#     area1 = cal_area(rect1)
#     area2 = cal_area(rect2)
#     print(area1+area2-cross)


# res=[]
# for line in input():
#     res.append(line.split())
# print(res)

# class Diycontextor:
#     def __init__(self, name, mode):
#         self.name = name
#         self.mode = mode
#
#     def __enter__(self):
#         print("Hi enter here!!")
#         self.filehander = open(self.name, self.mode)
#         return self.filehander
#
#     def __exit__(self, *para):
#         print( "Hi exit here")
#         self.filehander.close()
#
#
# with Diycontextor('py_ana.py', 'a+') as f:
#     for i in f:
#         print(i)
# res=[]
# for i in range(2):
#     Num = list(map(str, input().strip().split()))
#     res.append(Num)
# tem1 = dict()
# tem2 = dict()
# for item in res[0][0]:
#     tem1[item] =0 if (item not in tem1) else tem1[item]+1
# for item in res[1][0]:
#     tem2[item] =0 if (item not in tem2) else tem2[item]+1
# if tem1==tem2:
#     print(1)
# else:
#     print(0)
# n= int(input().strip())
# n= int(input().strip())
# # def count_prime(n):
# #     if n<7:
# #         return 0
# #     isPrime =list(range(n))
# #     isPrime[0]=isPrime[1]=0
# #     for i in range(2,int(n**0.5)+1):
# #         if isPrime[i]:
# #             isPrime[i*i:n:i] = [0]*((n-1-i*i)//i+1)
# #     isPrime=filter(lambda x:x!=0,isPrime)
# #     p=list(isPrime)
# #
# #     i ,j ,res =0,1,[]
# #     while p[j] <= n//2+1:
# #         cur_sum =sum(p[i:j+1])
# #         if cur_sum < n:
# #             j+=1
# #         elif cur_sum>n:
# #             i+=1
# #         else:
# #             res.append(p[i:j+1])
# #             j+=1
# #     flag =1
# #     for i in range(2, n):
# #         if n % i == 0:
# #             break
# #     else:
# #         flag-=1
# #     if flag==0:
# #         return len(res)+1
# #     else:
# #         return len(res)
# # print(count_prime(n))

# list = [1,2,3,4]
# print(next(iter(list)))
# for i in iter(list):
#     print(i)
#
# import csv
#
# with open('/Users/andy/Desktop/aic_origin_result.csv') as f:
#
#     f_csv = csv.reader(f)
#
#     headers = next(f_csv)
#
#     print(headers)
#
#     for row in f_csv:
#
#         print(row)# type:list

# for i in range(5):
#     print(i)
#     if i == 3:
#         break
# print(11111)

# s = input().strip()
# print(s)
# dic = {}
# for i in s:
#     if i in dic:
#         dic[i]+=1
#     else:
#         dic[i]=1
# #
# # print(dic['o'])
# n = len(s)
# ls = []
# for i in range(n):
#     if s[i] == 'G':
#         ls.append(s[i])
#         for j in range(i,n):
#             s[j]

# t = input().strip()
# s = 'Good'
# n,m = len(s),len(t)
# inx = 999999
# flag = 1
# while inx >= m-1:
#     if flag ==1:
#         inx=-1
#         flag-=1
#     for i in range(n):
#         for j in range(inx+1,m):
#             if t[j] == s[i]:
#                 inx = j
#                 if inx ==m:
#                     break
#                 print(inx)
#                 break
#         else:
#             print(0)
#     print(1)
# #
# a,b = [int(i) for i in input().split()]
# ls=[]
# for i in range(a):
#     Num = list(map(int, input().strip().split()))
#     ls.append(Num)
# if not ls:
#     print(0)
# x = len(ls)
# y = len(ls[0])
# dp = [[1 for _ in range(y)]for _ in range(x)]
# numsSort = sorted(sum([[(ls[i][j],i,j) for j in range(y)]for i in range(x)],[]))
# for i,j,k in numsSort:
#     dp[j][k] = 1+max(
#         dp[j-1][k] if j and ls[j-1][k]<i else 0,
#         dp[j][k-1] if k and ls[j][k-1] < i else 0,
#         dp[j + 1][k] if j!=x-1 and ls[j + 1][k] < i else 0,
#         dp[j ][k+1] if k!=y-1 and ls[j][k+1] < i else 0
#     )
# print(max(sum(dp,[])))
# # # n = int(sys.stdin.readline().strip())
# #     ans = 0
#     for i in range(a):
#         # 读取每一行
#         line = sys.stdin.readline().strip()
#         # 把每一行的数字分隔后转化成int列表
#         values = list(map(int, line.split()))
#         for v in values:
#             ans += v
#     print(ans)
# import sys
# n = int(sys.stdin.readline().strip())
# ans = 0
# for i in range(n):
#     # 读取每一行
#     line = sys.stdin.readline().strip()
#     # 把每一行的数字分隔后转化成int列表
#     values = list(map(int, line.split()))
#     for v in values:
#         ans += v
# print(ans)

# print(7//2)

# s = input().strip()
# ans = 0
# ans2=0
# count = 0
# count2=0
# for i in s:
#     if i =='(' or i ==')':
#         if i == '(':
#             ans+=1
#         else:
#             ans-=1
#         if ans<0:
#             count+=1
#             ans+=1
# count+=ans
# # print(count)
# for i in s:
#     if i =='[' or i ==']':
#         if i == '[':
#             ans2+=1
#         else:
#             ans2-=1
#         if ans2<0:
#             count2+=1
#             ans2+=1
# count2+=ans2
# #
# print(count+count2)


num = input().strip()
num = int(num)
ls =[]
def de(f,C,D):
    unit = (D-C)/10000
    s=0
    for i in range(10000):
        s+= f(C+unit*i)*unit
    return s
for i in range(num):
    Num = list(map(int, input().strip().split()))
    A,B,C,D= Num
    def f(x):
        return A*x**2+x+B
    # f = lambda x:A*x**2+x+B
    result=de(f,C,D)
    ls.append(result)
for i in range(len(ls)):
    print(ls[i])

# num = input().strip()
# num = int(num)
# if num<1 or num>10**9:
#     print(0)
# print((num*(2**(num-1)))%(10**9+7))

#
# s = input().strip()
# if not s:
#     print(0)
# else:
#     ans = 0
#     ans2=0
#     count = 0
#     count2=0
#     for i in s:
#         if i =='(' or i ==')':
#             if i == '(':
#                 ans+=1
#             else:
#                 ans-=1
#             if ans<0:
#                 count+=1
#                 ans+=1
#     count+=ans
#     for i in s:
#         if i =='[' or i ==']':
#             if i == '[':
#                 ans2+=1
#             else:
#                 ans2-=1
#             if ans2<0:
#                 count2+=1
#                 ans2+=1
#     count2+=ans2
#     print(count+count2)
# import
# num = input().strip()
# num = int(num)
