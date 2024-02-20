import os
import requests
from time import sleep
from bs4 import BeautifulSoup
import json
import random
import datetime
from requests.adapters import Retry
from dotenv import load_dotenv, set_key
import re
import sys
import pymysql
import uuid
from urllib.parse import urlparse, parse_qs

is_prod = not __file__[:-3].endswith("Dev")

if is_prod:
    load_dotenv()
    print('now in prod mode')
else:   
    load_dotenv('.env.dev')
    print('now in dev mode')

# config
startYear = os.getenv('START_YEAR')
# startMonth = os.getenv('START_MONTH')
# startDay = os.getenv('START_DAY')

endYear = os.getenv('END_YEAR')
# endMonth = os.getenv('END_MONTH')
# endDay = os.getenv('END_DAY')

jnCode = os.getenv('JN_CODE')
jnName = os.getenv('JN_NAME')

fetchType = os.getenv('FETCH_TYPE')

baseUrl = os.getenv('BASE_URL')
# downloadUrl = os.getenv('DOWNLOAD_URL')
cookie_str = os.getenv('COOKIE_STR')
referer = os.getenv('REFERER')

# 创建一个Retry对象，并设置重试次数为3
retry = Retry(total=50, backoff_factor=0.5)
# 创建一个Session对象，并将Retry对象应用于它
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

db_config = {
    "host": os.getenv('DB_HOST'),
    "user": os.getenv('DB_USERNAME'),
    "password": os.getenv('DB_PASSWORD'),
    "db": os.getenv('DB_DBNAME')
}

# 连接数据库
conn = pymysql.connect(**db_config)


cookies = {}
cookie_str = cookie_str.replace(" ", "")
pairs = cookie_str.split(";")
for pair in pairs:
    parts = pair.split("=", 1)
    key = parts[0].strip()
    value = parts[1].strip() if len(parts) > 1 else ""
    cookies[key] = value
# json_cookies = json.dumps(cookies)
# print(json_cookies)
# cookies = json_cookies

# all in one headers for 官方，否则 header 不全导致拉取不到
headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Cookie': cookie_str,
    'Origin': 'https://navi.cnki.net',
    'Pragma': 'no-cache',
    'Referer': referer,
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent':
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua':
    '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"'
}


def overwrite_env_variable(variable_name, new_value):
    if is_prod:
        set_key('.env', variable_name, str(new_value))
    else:   
        set_key('.env.dev', variable_name, str(new_value))


def check_anti_policy(content):
    # 这里通常会报：输入数字验证，一般 5 分钟后不需要输入验证码重试即可自动解冻继续访问。除下载以外这些不需要用户登录。
    # 拦截的原因是如果不拦截，则会一直正常执行下去，不抛异常则无法推到重启环节，导致程序”正常“结束。
    if content == '<script>window.location.href="/knavi/access/verification"</script>':
        print(
            "May trigger verification anti-policy. Exit and wait for restart.")
        sys.exit(-1)

def check_link_validity(link, endYear):
    parent_dl = link.find_parent("dl")
    if parent_dl:
        id_value = parent_dl.get("id", "")
        year_match = re.match(r"(\d{4})", id_value)
        if year_match:
            year = int(year_match.group(0))
            if year != int(endYear):
                return False
    return True

# 获取期刊列表
def get_jn_list(year):
    url = baseUrl + "/"+jnCode+"/yearList"
    data = {"pIdx": "0", "time": "PTsD9-wItUHENOfOCdc_syynG_ddT7_34uiGRZ28c_xGPXJq4WZpipBGDWx9fbwm0gUcaJmwBJYhUIopmuadKq-LNCRHaU3G"}
    response = session.post(url, headers=headers, data=data)
    # print(response.status_code)
    # print(response.content)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a")
        # 反扒判定，异常则强退等重启
        check_anti_policy(response.text)
        # print('get_jn_list response: ')
        # print(response.text[:9999])
        result = []
        for link in links:
            if check_link_validity(link, year):
                # print(link)
                result.append({"link": link["value"], "text": str(year) + ' ' + link.text})
        # print(result)
        return result

    else:
        print("Failed to get days.")
        return []


# 获取期刊内容列表
def get_papers(link):
    url = baseUrl + "/"+jnCode+"/papers"
    data = {
        "pcode": "CJFD,CCJD",
        "pageIdx": "0",
        "yearIssue": link
    }
    response = session.post(url, headers=headers, data=data)
    # print(data)
    if response.status_code == 200:
        # 反扒判定，异常则强退等重启
        check_anti_policy(response.text)
        # print('get_papers response: ')
        # print(response.text[:9999])
        return response
    else:
        print("Failed to get items.")
        return []

# 获取期刊详情信息
def get_paper_info(link):
    response = session.post(link, headers=headers)
    # print(data)
    if response.status_code == 200:
        # 反扒判定，异常则强退等重启
        check_anti_policy(response.text)
        # print('get_papers response: ')
        # print(response.text[:9999])
        return response
    else:
        print("Failed to get items.")
        return []

# 获取期刊引用详情
def get_paper_ref(link):
    parsed_url = urlparse(link)
    query_params = parse_qs(parsed_url.query)

    v_value = query_params.get('v')[0]  # Extract the value of 'v' parameter
    # print(v_value)
    response = session.get('https://kns.cnki.net/restapi/citation-api/v1/literature/references?v='+v_value+'&type=CJFDREF&start=1&size=50', headers=headers)
    # print(data)
    if response.status_code == 200:
        # 反扒判定，异常则强退等重启
        # TODO 此处在更新 ref 的时候，判定似乎不太对，得更新判定，否则一直拿到空，然后引导到 c8985580-b03f-44b9-8171-a5b399160637 里，造成全部都是 c8985580-b03f-44b9-8171-a5b399160637
        check_anti_policy(response.text)
        # print('get_papers response: ')
        # print(response.text[:9999])
        return response
    else:
        print("Failed to get items.")
        return []

def insert_paper_to_mysql(db_config, paper):
    # Connect to the MySQL database
    conn = pymysql.connect(**db_config)
    
    if paper['uuid'] and paper['title'] and paper['authors'] and  paper['year'] and paper['journal'] and paper['abstract']:
        try:
            with conn.cursor() as cursor:
                # SQL query for inserting data
                query = '''INSERT INTO papercollection (uuid, paper_title, authors, publication_year, journal_name, abstract) 
                        VALUES (%s ,%s, %s, %s, %s, %s)'''
                # Executing the query with the paper data
                cursor.execute(query, (paper['uuid'], paper['title'], paper['authors'], paper['year'], 
                                    paper['journal'], paper['abstract']))
                # Committing the transaction
                conn.commit()
        finally:
            # Closing the connection
            conn.close() 
    else:
        print('Paper data not complete, skip it.')

def get_paper_uuid(title, year, authors):
    try:
        uuid = ''
        sql = '''
            SELECT UUID 
            FROM papercollection 
            WHERE paper_title = %s AND publication_year = %s AND authors = %s
            '''
        # 连接数据库
        conn = pymysql.connect(**db_config)
        with conn.cursor() as cursor:
            # 执行SQL查询
            cursor.execute(sql, (title, year, authors))
            # 获取查询结果
            result = cursor.fetchone()  # 假设只有一条记录匹配
            if result:
                uuid = result[0]
                print("Found UUID:", result[0])
            else:
                print("No records found.")
    finally:
        # 关闭数据库连接
        conn.close()
        return uuid  

def check_paper_exists(db_config, title, year, authors):
    """检查数据库中是否已存在相同的论文记录"""
    # 连接数据库
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            # SQL查询语句
            sql = '''
            SELECT COUNT(*) FROM papercollection
            WHERE paper_title = %s AND publication_year = %s AND authors = %s
            '''
            cursor.execute(sql, (title, year, authors))
            (count,) = cursor.fetchone()
            return count > 0
    finally:
        conn.close()
        
# 主函数
def main():
    for year in range(int(startYear), int(endYear)-1, -1):  # 循环从startYear年到2020年
        print('start year: ' + str(year))
        links = get_jn_list(year)
        # print('get links: ')
        # print(links)
        
        for link in links:
            response = get_papers(link['link'])
            soup = BeautifulSoup(response.text, "html.parser")
            papers = soup.select("span.name > a")
            # print(response.text)
            # print(uls)
    
            if not papers:
                continue

            for paper in papers:

                print('start days at: ' + link['text'])
                # 在这里处理找到的第一个li标签

                paper_link = paper['href']
                print('paper_link: ' + paper_link)
                
                
                if fetchType == 'PAPER':
                    response = get_paper_info(paper_link)
                    # print(response)
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    
                    print('paper_year:'+str(year))
                    print('paper_jn:'+jnName)
                    
                    paper_title_element = soup.find("h1")
                    if paper_title_element:
                        paper_title = paper_title_element.text
                        print('paper_title:'+paper_title)
                    else:
                        print('Ttile not found, skip it.')
                    
                    paper_authors = soup.select(".author a")
                    author_texts = [author.text for author in paper_authors]
                    authors_combined = ", ".join(author_texts)
                    if authors_combined:
                        print('author_texts:'+authors_combined)
                    else:
                        print('Authors not found, skip it.')
                                            
                    paper_abstract_element = soup.find("span", id="ChDivSummary")
                    if paper_abstract_element:
                        paper_abstract = paper_abstract_element.text
                        print('paper_abstract:'+paper_abstract)
                    else:
                        print('Abstract not found, skip it.')
                        
                    if paper_title and paper_abstract and authors_combined:
                        # 检测是否已经插入过此条记录
                        if not check_paper_exists(db_config, paper_title, year, authors_combined):
                            paper_data = {
                                "uuid": str(uuid.uuid4()),
                                "title": paper_title,
                                "authors": authors_combined,
                                "year": year,
                                "journal": jnName,
                                "abstract": paper_abstract
                            }
                            insert_paper_to_mysql(db_config, paper_data)
                        else:
                            print('Paper 已录入过, skip it.')

                if fetchType == 'REFERENCE':
                    
                    # STEP 1 拿到当前文章 uuid（必须先跑过 FETCH_TYPE = 'PAPER' 流程后再进行本 REFERENCE 流程）
                    response = get_paper_info(paper_link)
                    # print(response)
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    paper_title_element = soup.find("h1")
                    if paper_title_element:
                        paper_title = paper_title_element.text
                        print('paper_title:'+paper_title)
                    else:
                        print('Ttile not found, skip it.')
                    
                    paper_authors = soup.select(".author a")
                    author_texts = [author.text for author in paper_authors]
                    authors_combined = ", ".join(author_texts)
                    if authors_combined:
                        print('author_texts:'+authors_combined)
                    else:
                        print('Authors not found, skip it.')
                        
                    cur_paper_uuid = ''
                    if paper_title and year and authors_combined:
                        cur_paper_uuid = get_paper_uuid(paper_title, year, authors_combined)
                        
                    if not cur_paper_uuid:
                        print('未找到此文章，一般不会发生此提示，可能之前采集的文章没有正常录入或此文章为更新的文章。本地引用录入跳过')
                    else:
                        response = get_paper_ref(paper_link)
                        response_data = json.loads(response.text)
                                                
                        for ref_paper in response_data['data']['data']:
                            
                            ref_paper_meta_title = ''
                            ref_paper_meta_authors = ''
                            ref_paper_meta_year = ''
                            ref_paper_meta_journal = ''
                            ref_paper_meta_abstract = ''
                            
                            for ref_paper_meta in ref_paper['metadata']:
                                # print('ref_paper_meta')
                                # print(ref_paper_meta)
                                
                                if ref_paper_meta['name'] == 'TI':
                                    ref_paper_meta_title = ref_paper_meta['value']
                                if ref_paper_meta['name'] == 'AU':
                                    ref_paper_meta_authors = ref_paper_meta['value'].replace(';',', ')
                                    # 移除末尾的逗号加空格
                                    if ref_paper_meta_authors.endswith(', '):
                                        ref_paper_meta_authors = ref_paper_meta_authors[:-2]
                                if ref_paper_meta['name'] == 'YE':
                                    ref_paper_meta_year = ref_paper_meta['value']
                                if ref_paper_meta['name'] == 'LY':
                                    ref_paper_meta_journal = ref_paper_meta['value']
                                if ref_paper_meta['name'] == 'AB':
                                    ref_paper_meta_abstract = ref_paper_meta['value']
                                 
                            # STEP 2 查询引用文章是否被收录过，如果用直接返，如果没有先插入再返回
                            if cur_paper_uuid:
                                has_paper_uuid = get_paper_uuid(ref_paper_meta_title, ref_paper_meta_year, ref_paper_meta_authors)
                                ref_paper_uuid = has_paper_uuid if has_paper_uuid else str(uuid.uuid4())
                                
                                ref_paper_data = {
                                        "uuid": ref_paper_uuid,
                                        "title": ref_paper_meta_title,
                                        "authors": ref_paper_meta_authors,
                                        "year": ref_paper_meta_year,
                                        "journal": ref_paper_meta_journal,
                                        "abstract": ref_paper_meta_abstract
                                    }

                                print('ref_paper_data')
                                print(ref_paper_data)
                                
                                # print(ref_paper_meta)
                                if ref_paper_meta_title and ref_paper_meta_year and ref_paper_meta_authors and ref_paper_meta_journal and ref_paper_meta_abstract:
                                    
                                    if not has_paper_uuid:
                                        insert_paper_to_mysql(db_config, ref_paper_data)
                                    
                                    # STEP 3 拿着当前 UUID，以及对方文章 UUID，插入引用关系表
                                    try:
                                        # 连接数据库
                                        conn = pymysql.connect(**db_config)
                                        with conn.cursor() as cursor:
                                            
                                            print('开始添加引用信息:')
                                            
                                            print('cur_paper_uuid:'+cur_paper_uuid)
                                            print('ref_paper_uuid:'+ref_paper_uuid)
                                            
                                        
                                            # # 外键预检：检查paper_uuid是否存在于papercollection表中
                                            # sql_check_paper_uuid = "SELECT COUNT(*) FROM papercollection WHERE uuid = %s"
                                            # cursor.execute(sql_check_paper_uuid, (cur_paper_uuid,))
                                            # if cursor.fetchone()[0] == 0:
                                            #     print("paper_uuid does not exist in papercollection.")
                                            #     # 这里可以添加逻辑来处理这种情况，例如插入新记录到papercollection或者跳过插入操作

                                            # # 类似地，检查referenced_paper_uuid
                                            # cursor.execute(sql_check_paper_uuid, (ref_paper_uuid,))
                                            # if cursor.fetchone()[0] == 0:
                                            #     print("referenced_paper_uuid does not exist in papercollection.")
                                            #     # 处理这种情况
                                        
                                            # 检查是否已存在记录
                                            sql_check = '''
                                            SELECT COUNT(*) FROM paperreferences 
                                            WHERE paper_uuid = %s AND referenced_paper_uuid = %s
                                            '''
                                            cursor.execute(sql_check, (cur_paper_uuid, ref_paper_uuid))
                                            (count,) = cursor.fetchone()
                                            
                                            # 如果不存在，则插入
                                            if count == 0:
                                                sql_insert = '''
                                                INSERT INTO paperreferences (paper_uuid, referenced_paper_uuid) 
                                                VALUES (%s, %s)
                                                '''
                                                cursor.execute(sql_insert, (cur_paper_uuid, ref_paper_uuid))
                                                conn.commit()
                                                print("Record inserted@paperreferences.")
                                            else:
                                                print("Record already exists@paperreferences. Skipping insert.")
                                    finally:
                                        conn.close() 
                                else:
                                    print('Reference paper data not complete, skip it. 可能存在原始 paper 为空值的情况，需要检查原始 paper 数据是否完整。')
        
                            else:
                                print('cur_paper_uuid not found, skip it.')
                random_time = random.randint(1, 1)
                sleep(random_time)            

            # 更新系统变量（前期大批量采集的时候用，用于倒序更新时，给当前采集点打点，直到其进度，也方便重启进程后继续采集）
            overwrite_env_variable('START_YEAR', year)
        

if __name__ == "__main__":
    main()
