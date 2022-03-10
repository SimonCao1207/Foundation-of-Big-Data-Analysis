import re
my_file = open("pg100.txt", "r")
content = my_file. read()
content_list = re.split(r'[^\w]+', content)
print(content_list)
