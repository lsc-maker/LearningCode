import requests
import time

def getHTMLText(url):
	try:
		r = requests.get(url,timeout=30)
		r.raise_for_status()
		r.enconding = r.apparent_encoding
		return r.text
	except:
		return "出现异常"

def main():
    url="http://www.baidu.com"
    i = 0
    start = time.perf_counter()
    while i < 100:
    	c = getHTMLText(url)
    	if c == '出现异常':
    		print("第{}次循环".format(i,c))
    		break
    	i += 1
    	end = time.perf_counter()
    	print('爬取次数：{},运行时间：{}'.format(i,end - start))

if __name__ == '__main__':
	main()
