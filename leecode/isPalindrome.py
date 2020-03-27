#判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
def isPalindrome(x: int) -> bool:
    if x < 0:
        return False
    else:
        y = str(x)[::-1]
        if y == str(x):
            return True
        else: 
            return False
print(isPalindrome(x=eval(input("请输入一个整数:"))))