import requests
keyword = "Python"
try:
	kv = {'q':keyword}
	r = requests.get("https://www.so.com/s",params=kv)
	print(r.request.url)
	r.raise_for_status()
	print(len(r.text))
except:
	print("爬取失败")