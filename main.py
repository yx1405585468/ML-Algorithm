def solution(nums):
    slow = 2
    for fast in range(2, len(nums)):
        if nums[fast] != nums[slow - 2]:
            nums[slow] = nums[fast]
            slow += 1
    print(nums)


if __name__ == '__main__':
    nums = [1, 1, 1, 2, 2, 3]
    solution(nums)
