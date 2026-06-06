"""
数据生成模块：基于正确Python函数 + AST bug注入，生成bug-fix代码对
5种缺陷类型：off_by_one, wrong_operator, wrong_variable, missing_condition, missing_statement
"""
import ast
import copy
import json
import os
import random
import sys

import config

# ============ 缺陷类型定义 ============
DEFECT_TYPES = [
    "off_by_one",
    "wrong_operator",
    "wrong_variable",
    "missing_condition",
    "missing_statement",
]
DEFECT_TYPE_TO_ID = {t: i for i, t in enumerate(DEFECT_TYPES)}


# ============ 正确Python函数库 ============
CORRECT_FUNCTIONS = [
    # --- 排序 ---
    ("bubble_sort", """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr"""),
    ("insertion_sort", """def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr"""),
    ("selection_sort", """def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr"""),
    # --- 搜索 ---
    ("binary_search", """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1"""),
    ("linear_search", """def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1"""),
    # --- 数学 ---
    ("factorial", """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""),
    ("fibonacci", """def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b"""),
    ("gcd", """def gcd(a, b):
    while b:
        a, b = b, a % b
    return a"""),
    ("lcm", """def lcm(a, b):
    return abs(a * b) // gcd(a, b)"""),
    ("is_prime", """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True"""),
    ("power", """def power(base, exp):
    if exp == 0:
        return 1
    result = 1
    for _ in range(exp):
        result *= base
    return result"""),
    ("count_divisors", """def count_divisors(n):
    count = 0
    for i in range(1, n + 1):
        if n % i == 0:
            count += 1
    return count"""),
    ("sum_of_digits", """def sum_of_digits(n):
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total"""),
    ("reverse_number", """def reverse_number(n):
    rev = 0
    while n > 0:
        rev = rev * 10 + n % 10
        n //= 10
    return rev"""),
    # --- 字符串 ---
    ("is_palindrome", """def is_palindrome(s):
    s = s.lower()
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True"""),
    ("count_vowels", """def count_vowels(s):
    vowels = 'aeiouAEIOU'
    count = 0
    for ch in s:
        if ch in vowels:
            count += 1
    return count"""),
    ("reverse_string", """def reverse_string(s):
    result = ''
    for i in range(len(s) - 1, -1, -1):
        result += s[i]
    return result"""),
    ("is_anagram", """def is_anagram(s1, s2):
    return sorted(s1.lower()) == sorted(s2.lower())"""),
    ("remove_char", """def remove_char(s, ch):
    result = ''
    for c in s:
        if c != ch:
            result += c
    return result"""),
    ("count_occurrences", """def count_occurrences(s, sub):
    count = 0
    i = 0
    while i <= len(s) - len(sub):
        if s[i:i+len(sub)] == sub:
            count += 1
            i += len(sub)
        else:
            i += 1
    return count"""),
    # --- 列表 ---
    ("find_max", """def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for val in arr[1:]:
        if val > max_val:
            max_val = val
    return max_val"""),
    ("find_min", """def find_min(arr):
    if not arr:
        return None
    min_val = arr[0]
    for val in arr[1:]:
        if val < min_val:
            min_val = val
    return min_val"""),
    ("remove_duplicates", """def remove_duplicates(arr):
    result = []
    seen = set()
    for item in arr:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result"""),
    ("flatten_list", """def flatten_list(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result"""),
    ("merge_sorted", """def merge_sorted(a, b):
    result = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result"""),
    ("find_second_largest", """def find_second_largest(arr):
    if len(arr) < 2:
        return None
    largest = max(arr[0], arr[1])
    second = min(arr[0], arr[1])
    for val in arr[2:]:
        if val > largest:
            second = largest
            largest = val
        elif val > second and val != largest:
            second = val
    return second"""),
    ("rotate_list", """def rotate_list(arr, k):
    if not arr:
        return arr
    k = k % len(arr)
    return arr[-k:] + arr[:-k]"""),
    # --- 栈/队列 ---
    ("is_valid_parentheses", """def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for ch in s:
        if ch in mapping:
            top = stack.pop() if stack else '#'
            if mapping[ch] != top:
                return False
        else:
            stack.append(ch)
    return not stack"""),
    ("evaluate_postfix", """def evaluate_postfix(expr):
    stack = []
    for token in expr.split():
        if token in '+-*/':
            b = stack.pop()
            a = stack.pop()
            if token == '+': stack.append(a + b)
            elif token == '-': stack.append(a - b)
            elif token == '*': stack.append(a * b)
            elif token == '/': stack.append(a // b)
        else:
            stack.append(int(token))
    return stack[0]"""),
    # --- 树 ---
    ("tree_height", """def tree_height(root):
    if root is None:
        return 0
    left_h = tree_height(root.left)
    right_h = tree_height(root.right)
    return max(left_h, right_h) + 1"""),
    ("tree_size", """def tree_size(root):
    if root is None:
        return 0
    return 1 + tree_size(root.left) + tree_size(root.right)"""),
    ("is_bst", """def is_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if root is None:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return is_bst(root.left, min_val, root.val) and is_bst(root.right, root.val, max_val)"""),
    # --- 动态规划 ---
    ("climbing_stairs", """def climbing_stairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b"""),
    ("max_subarray_sum", """def max_subarray_sum(arr):
    max_sum = arr[0]
    current_sum = arr[0]
    for val in arr[1:]:
        current_sum = max(val, current_sum + val)
        max_sum = max(max_sum, current_sum)
    return max_sum"""),
    ("knapsack", """def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]"""),
    ("longest_increasing_subsequence", """def longest_increasing_subsequence(arr):
    if not arr:
        return 0
    dp = [1] * len(arr)
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)"""),
    ("coin_change", """def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1"""),
    # --- 图 ---
    ("bfs", """def bfs(graph, start):
    visited = set()
    queue = [start]
    visited.add(start)
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return result"""),
    ("dfs", """def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))
    return result"""),
    ("has_cycle", """def has_cycle(graph):
    visited = set()
    rec_stack = set()
    def visit(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if visit(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False
    for node in graph:
        if node not in visited:
            if visit(node):
                return True
    return False"""),
    # --- 工具 ---
    ("two_sum", """def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []"""),
    ("roman_to_int", """def roman_to_int(s):
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    for i in range(len(s)):
        if i + 1 < len(s) and values[s[i]] < values[s[i+1]]:
            total -= values[s[i]]
        else:
            total += values[s[i]]
    return total"""),
    ("int_to_roman", """def int_to_roman(num):
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    sym = ['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
    result = ''
    for i in range(len(val)):
        while num >= val[i]:
            result += sym[i]
            num -= val[i]
    return result"""),
    ("is_armstrong", """def is_armstrong(n):
    digits = str(n)
    power = len(digits)
    total = sum(int(d) ** power for d in digits)
    return total == n"""),
    ("matrix_multiply", """def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result"""),
    ("transpose_matrix", """def transpose_matrix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]"""),
    # --- 更多数学 ---
    ("count_primes", """def count_primes(n):
    if n < 2:
        return 0
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n, i):
                is_prime[j] = False
    return sum(is_prime)"""),
    ("perfect_number", """def perfect_number(n):
    if n < 2:
        return False
    total = 1
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
    return total == n"""),
    ("fibonacci_sum", """def fibonacci_sum(n):
    a, b = 0, 1
    total = 0
    while a <= n:
        total += a
        a, b = b, a + b
    return total"""),
    # --- 更多列表操作 ---
    ("chunk_list", """def chunk_list(arr, size):
    result = []
    for i in range(0, len(arr), size):
        result.append(arr[i:i+size])
    return result"""),
    ("interleave_lists", """def interleave_lists(a, b):
    result = []
    min_len = min(len(a), len(b))
    for i in range(min_len):
        result.append(a[i])
        result.append(b[i])
    result.extend(a[min_len:])
    result.extend(b[min_len:])
    return result"""),
    ("pair_sum", """def pair_sum(arr, target):
    pairs = []
    seen = set()
    for num in arr:
        complement = target - num
        if complement in seen:
            pairs.append((min(num, complement), max(num, complement)))
        seen.add(num)
    return pairs"""),
    # --- 更多字符串 ---
    ("camel_to_snake", """def camel_to_snake(s):
    result = ''
    for ch in s:
        if ch.isupper():
            result += '_' + ch.lower()
        else:
            result += ch
    return result.lstrip('_')"""),
    ("most_frequent_char", """def most_frequent_char(s):
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    max_char = ''
    max_count = 0
    for ch, count in freq.items():
        if count > max_count:
            max_count = count
            max_char = ch
    return max_char"""),
    # --- 额外算法 ---
    ("quick_sort", """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)"""),
    ("merge_sort", """def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge_sorted(left, right)"""),
    ("heapify", """def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)"""),
    ("heap_sort", """def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    return arr"""),
    # --- 更多搜索 ---
    ("find_peak", """def find_peak(arr):
    for i in range(len(arr)):
        left = arr[i-1] if i > 0 else float('-inf')
        right = arr[i+1] if i < len(arr)-1 else float('-inf')
        if arr[i] >= left and arr[i] >= right:
            return i
    return -1"""),
    ("count_negatives", """def count_negatives(grid):
    count = 0
    for row in grid:
        for val in row:
            if val < 0:
                count += 1
    return count"""),
    # --- 更多工具 ---
    ("valid_ip", """def valid_ip(s):
    parts = s.split('.')
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        num = int(part)
        if num < 0 or num > 255:
            return False
        if len(part) > 1 and part[0] == '0':
            return False
    return True"""),
    ("title_case", """def title_case(s):
    words = s.split()
    result = []
    for word in words:
        if word:
            result.append(word[0].upper() + word[1:].lower())
        else:
            result.append(word)
    return ' '.join(result)"""),
    ("run_length_encoding", """def run_length_encoding(s):
    if not s:
        return ''
    result = ''
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            result += s[i-1] + str(count)
            count = 1
    result += s[-1] + str(count)
    return result"""),
    ("pascal_triangle", """def pascal_triangle(n):
    triangle = []
    for i in range(n):
        row = [1] * (i + 1)
        for j in range(1, i):
            row[j] = triangle[i-1][j-1] + triangle[i-1][j]
        triangle.append(row)
    return triangle"""),
    # --- 位操作 ---
    ("count_set_bits", """def count_set_bits(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count"""),
    ("is_power_of_two", """def is_power_of_two(n):
    if n <= 0:
        return False
    return (n & (n - 1)) == 0"""),
    ("reverse_bits", """def reverse_bits(n, bits=32):
    result = 0
    for _ in range(bits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result"""),
]


# ============ AST Bug注入器 ============
class OffByOneInjector(ast.NodeTransformer):
    """注入off-by-one错误：修改range边界和比较运算符"""
    def __init__(self):
        self.modified = False

    def visit_Compare(self, node):
        self.generic_visit(node)
        if not self.modified and node.ops:
            for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
                if isinstance(op, ast.Lt):
                    node.ops[i] = ast.LtE()
                    self.modified = True
                    break
                elif isinstance(op, ast.Gt):
                    node.ops[i] = ast.GtE()
                    self.modified = True
                    break
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if not self.modified and isinstance(node.func, ast.Name) and node.func.id == 'range':
            if len(node.args) >= 1:
                last_arg = node.args[-1]
                if isinstance(last_arg, ast.BinOp) and isinstance(last_arg.op, ast.Sub):
                    node.args[-1] = ast.BinOp(left=last_arg.left, op=ast.Add(), right=last_arg.right)
                    self.modified = True
                elif isinstance(last_arg, ast.Name) or isinstance(last_arg, ast.Constant):
                    node.args[-1] = ast.BinOp(left=last_arg, op=ast.Add(), right=ast.Constant(value=1))
                    self.modified = True
        return node


class WrongOperatorInjector(ast.NodeTransformer):
    """注入运算符错误：替换二元运算符"""
    OP_MAP = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
        ast.Mult: ast.Div,
        ast.Div: ast.Mult,
    }

    def __init__(self):
        self.modified = False

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if not self.modified and type(node.op) in self.OP_MAP:
            node.op = self.OP_MAP[type(node.op)]()
            self.modified = True
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if not self.modified:
            if isinstance(node.op, ast.And):
                node.op = ast.Or()
                self.modified = True
            elif isinstance(node.op, ast.Or):
                node.op = ast.And()
                self.modified = True
        return node


class WrongVariableInjector(ast.NodeTransformer):
    """注入变量名错误：替换变量名为同作用域其他变量"""
    def __init__(self):
        self.modified = False
        self.scope_vars = set()

    def visit_FunctionDef(self, node):
        # 收集函数内所有变量名
        self.scope_vars = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                self.scope_vars.add(child.id)
        self.scope_vars = list(self.scope_vars)
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if not self.modified and isinstance(node.ctx, ast.Load) and len(self.scope_vars) >= 2:
            candidates = [v for v in self.scope_vars if v != node.id]
            if candidates:
                node.id = random.choice(candidates)
                self.modified = True
        return node


class MissingConditionInjector(ast.NodeTransformer):
    """注入缺少条件错误：移除if判断或简化条件"""
    def __init__(self):
        self.modified = False

    def visit_If(self, node):
        self.generic_visit(node)
        if not self.modified and node.body and not node.orelse:
            # 只有无if-else的简单条件才移除，将if body直接提升
            parent_replacement = node.body
            self.modified = True
            return parent_replacement
        return node


class MissingStatementInjector(ast.NodeTransformer):
    """注入缺少语句错误：移除一条语句"""
    def __init__(self):
        self.modified = False

    def visit_FunctionDef(self, node):
        new_body = []
        removed = False
        for stmt in node.body:
            if not removed and not isinstance(stmt, ast.Return) and len(node.body) > 1:
                # 跳过一条非return语句
                removed = True
                self.modified = True
                continue
            new_body.append(stmt)
        if removed:
            node.body = new_body
        return node


INJECTORS = {
    "off_by_one": OffByOneInjector,
    "wrong_operator": WrongOperatorInjector,
    "wrong_variable": WrongVariableInjector,
    "missing_condition": MissingConditionInjector,
    "missing_statement": MissingStatementInjector,
}


def inject_bug(code, defect_type):
    """对正确代码注入指定类型的bug，返回buggy代码（若注入失败返回None）"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    injector = INJECTORS[defect_type]()
    try:
        new_tree = injector.visit(tree)
        ast.fix_missing_locations(new_tree)
        if not injector.modified:
            return None
        buggy_code = ast.unparse(new_tree)
        # 验证buggy代码可以被解析
        ast.parse(buggy_code)
        return buggy_code
    except Exception:
        return None


def generate_dataset():
    """生成完整的bug-fix数据集"""
    random.seed(config.RANDOM_SEED)
    dataset = []

    for func_name, correct_code in CORRECT_FUNCTIONS:
        for defect_type in DEFECT_TYPES:
            buggy_code = inject_bug(correct_code, defect_type)
            if buggy_code is not None and buggy_code != correct_code:
                dataset.append({
                    "function_name": func_name,
                    "buggy_code": buggy_code,
                    "fixed_code": correct_code,
                    "defect_type": defect_type,
                    "defect_type_id": DEFECT_TYPE_TO_ID[defect_type],
                })

    return dataset


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    dataset = generate_dataset()

    # 按function_name分组，确保train/test不泄漏
    func_names = sorted(set(d["function_name"] for d in dataset))
    random.shuffle(func_names)
    split_idx = int(len(func_names) * (1 - config.TEST_RATIO))
    train_funcs = set(func_names[:split_idx])
    test_funcs = set(func_names[split_idx:])

    train_data = [d for d in dataset if d["function_name"] in train_funcs]
    test_data = [d for d in dataset if d["function_name"] in test_funcs]

    for d in train_data:
        d["split"] = "train"
    for d in test_data:
        d["split"] = "test"

    # 保存
    all_data = train_data + test_data
    with open(config.DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    # 统计
    print(f"数据集生成完成:")
    print(f"  总样本数: {len(all_data)}")
    print(f"  训练集: {len(train_data)}")
    print(f"  测试集: {len(test_data)}")
    print(f"  缺陷类型分布:")
    for dt in DEFECT_TYPES:
        count = sum(1 for d in all_data if d["defect_type"] == dt)
        print(f"    {dt}: {count}")


if __name__ == "__main__":
    main()
