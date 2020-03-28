import requests
text="202.204.80.112"
url = "https://m.ip138.com/iplookup.asp?ip={}".format(text)
headers = {"user-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; X64) AppleWebKit/537.36 (KHTML,like Gecko) Chrome/80.0.3987.100 Safari/537.36",}
try:
	r = requests.get(url,headers=headers)
	r.raise_for_status()
	r.encooding = r.apparent_encoding
	print(r.text)
except:
	print('爬取失败')