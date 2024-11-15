import time
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
import sys

# Set a large number to represent infinity in Python
INT_MAX = sys.maxsize

# Program 3
class Painting:
    def __init__(self, height, width):
        self.height = height
        self.width = width

class Platform:
    def __init__(self, max_allowed_width):
        self.width = 0
        self.height = 0
        self.max_allowed_width = max_allowed_width
        self.paintings = []

    def painting_fit(self, painting: Painting) -> bool:
        return self.width + painting.width <= self.max_allowed_width

    def add_painting(self, painting: Painting):
        self.paintings.append(painting)
        self.width += painting.width
        self.height = max(self.height, painting.height)

def program3(n: int, W: int, heights: List[int], widths: List[int]) -> Tuple[int, int, List[int]]:
    optimal_total_height = INT_MAX
    optimal_partition_sizes = []
    paintings = [Painting(heights[i], widths[i]) for i in range(n)]

    for i in range(1 << (n - 1)):
        platforms = []
        platform_sizes = []
        current_platform = Platform(W)
        skip = False

        for j in range(n):
            if not current_platform.painting_fit(paintings[j]):
                skip = True
                break
            current_platform.add_painting(paintings[j])
            if j == n - 1 or (i & (1 << j)):
                platform_sizes.append(len(current_platform.paintings))
                platforms.append(current_platform)
                current_platform = Platform(W)

        if not skip:
            platform_height_sum = sum(platform.height for platform in platforms)
            if platform_height_sum < optimal_total_height:
                optimal_total_height = platform_height_sum
                optimal_partition_sizes = platform_sizes

    return len(optimal_partition_sizes), optimal_total_height, optimal_partition_sizes

# Program 4
class PlatformDP:
    def __init__(self, height=INT_MAX, number_of_paintings=0, prev_platform_end=-1):
        self.height = height
        self.number_of_paintings = number_of_paintings
        self.prev_platform_end = prev_platform_end

def program4(n: int, W: int, heights: List[int], widths: List[int]) -> Tuple[int, int, List[int]]:
    OPT = [[PlatformDP() for _ in range(n + 1)] for _ in range(n + 1)]
    platform_sizes = []
    dp = [PlatformDP(0, 0, -1) for _ in range(n + 1)]

    for i in range(n):
        for j in range(i, n):
            total_width = 0
            max_height = 0
            for k in range(i, j + 1):
                total_width += widths[k]
                max_height = max(max_height, heights[k])
                if total_width > W:
                    OPT[i][j] = PlatformDP()
                    break
            if total_width <= W:
                OPT[i][j] = PlatformDP(max_height, j - i + 1, -1)

    for i in range(1, n + 1):
        for j in range(i):
            if OPT[j][i - 1].height != INT_MAX:
                new_height = dp[j].height + OPT[j][i - 1].height
                if new_height < dp[i].height:
                    dp[i] = PlatformDP(new_height, 1, j)

    sum_optimal_height = dp[n].height
    curr = n
    while curr > 0:
        platform_sizes.append(curr - dp[curr].prev_platform_end)
        curr = dp[curr].prev_platform_end

    platform_sizes.reverse()
    return len(platform_sizes), sum_optimal_height, platform_sizes

# Program 5A
class DP:
    def __init__(self, total_height=INT_MAX, prev_start=-1):
        self.total_height = total_height
        self.prev_start = prev_start

def program5A(n: int, W: int, heights: List[int], widths: List[int]) -> Tuple[int, int, List[int]]:
    paintings = [Painting(heights[i], widths[i]) for i in range(n)]
    OPT = [DP() for _ in range(n + 1)]
    OPT[0].total_height = 0

    for i in range(1, n + 1):
        current_platform = Platform(W)
        for j in range(i - 1, -1, -1):
            if not current_platform.painting_fit(paintings[j]):
                break
            current_platform.add_painting(paintings[j])
            total_height = OPT[j].total_height + current_platform.height
            if total_height < OPT[i].total_height:
                OPT[i].total_height = total_height
                OPT[i].prev_start = j

    platform_sizes = []
    current = n
    while current > 0:
        platform_sizes.append(current - OPT[current].prev_start)
        current = OPT[current].prev_start

    platform_sizes.reverse()
    return len(platform_sizes), OPT[n].total_height, platform_sizes

# Program 5B
def program5B(n: int, W: int, heights: List[int], widths: List[int]) -> Tuple[int, int, List[int]]:
    paintings = [Painting(heights[i], widths[i]) for i in range(n)]
    OPT = [DP() for _ in range(n + 1)]
    OPT[0].total_height = 0

    for i in range(1, n + 1):
        current_platform = Platform(W)
        for j in range(i - 1, -1, -1):
            if not current_platform.painting_fit(paintings[j]):
                break
            current_platform.add_painting(paintings[j])
            total_height = OPT[j].total_height + current_platform.height
            if total_height < OPT[i].total_height:
                OPT[i].total_height = total_height
                OPT[i].prev_start = j

    platform_sizes = []
    current = n
    while current > 0:
        platform_sizes.append(current - OPT[current].prev_start)
        current = OPT[current].prev_start

    platform_sizes.reverse()
    return len(platform_sizes), OPT[n].total_height, platform_sizes

# Define a function to measure execution time of a given function
def measure_runtime(func, *args):
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time

# Values for n to test
n_values = [100, 200, 300, 400, 500]
W = 10  # Arbitrary max width for platform
heights = [random.randint(1, 10) for _ in range(max(n_values))]
widths = [random.randint(1, 10) for _ in range(max(n_values))]

# Lists to store runtimes
program3_runtimes = []
program4_runtimes = []
program5A_runtimes = []
program5B_runtimes = []

# Measure runtime for each n value for all programs
for n in n_values:
    program3_runtimes.append(measure_runtime(program3, n, W, heights[:n], widths[:n]))
    program4_runtimes.append(measure_runtime(program4, n, W, heights[:n], widths[:n]))
    program5A_runtimes.append(measure_runtime(program5A, n, W, heights[:n], widths[:n]))
    program5B_runtimes.append(measure_runtime(program5B, n, W, heights[:n], widths[:n]))

# Plot individual runtimes
plt.figure(figsize=(10, 6))
plt.plot(n_values, program3_runtimes, label="Program 3", marker='o')
plt.xlabel('n')
plt.ylabel('Runtime (s)')
plt.title('Runtime of Program 3')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(n_values, program4_runtimes, label="Program 4", marker='o')
plt.xlabel('n')
plt.ylabel('Runtime (s)')
plt.title('Runtime of Program 4')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(n_values, program5A_runtimes, label="Program 5A", marker='o')
plt.xlabel('n')
plt.ylabel('Runtime (s)')
plt.title('Runtime of Program 5A')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(n_values, program5B_runtimes, label="Program 5B", marker='o')
plt.xlabel('n')
plt.ylabel('Runtime (s)')
plt.title('Runtime of Program 5B')
plt.legend()
plt.show()

# Combined runtime plot
plt.figure(figsize=(10, 6))
plt.plot(n_values, program3_runtimes, label="Program 3", marker='o')
plt.plot(n_values, program4_runtimes, label="Program 4", marker='o')
plt.plot(n_values, program5A_runtimes, label="Program 5A", marker='o')
plt.plot(n_values, program5B_runtimes, label="Program 5B", marker='o')
plt.xlabel('n')
plt.ylabel('Runtime (s)')
plt.title('Runtimes of All Programs')
plt.legend()
plt.show()

# Plot comparison between 5A and 5B
plt.figure(figsize=(10, 6))
plt.plot(n_values, program5A_runtimes, label="Program 5A", marker='o')
plt.plot(n_values, program5B_runtimes, label="Program 5B", marker='o')
plt.xlabel('n')
plt.ylabel('Runtime (s)')
plt.title('Runtime Comparison: Program 5A vs Program 5B')
plt.legend()
plt.show()
