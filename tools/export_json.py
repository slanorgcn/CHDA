import pymysql
import json
from dotenv import load_dotenv, set_key

load_dotenv()

# 数据库连接配置
db_config = {
    "host": os.getenv('DB_HOST'),
    "user": os.getenv('DB_USERNAME'),
    "password": os.getenv('DB_PASSWORD'),
    "db": os.getenv('DB_DBNAME')
}

# 连接数据库
conn = pymysql.connect(**db_config)

try:
    papers = []
    edges = []
    with conn.cursor() as cursor:
        # 查询所有论文
        cursor.execute("SELECT uuid, paper_title, publication_year, journal_name, authors, abstract FROM papercollection WHERE is_deleted = 0")
        for row in cursor.fetchall():
            uuid, title, year, journal, authors, abstract = row
            paper = {
                "id": uuid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "authors": authors.split(', ')
            }
            papers.append(paper)

        # 查询所有引用关系
        cursor.execute("SELECT paper_uuid, referenced_paper_uuid FROM paperreferences WHERE is_deleted = 0")
        for row in cursor.fetchall():
            source, target = row
            # 查找是否已存在相同的源节点
            edge = next((e for e in edges if e['source'] == source), None)
            if edge:
                edge['target'].append(target)
            else:
                edges.append({
                    "source": source,
                    "target": [target]
                })

    # 整合数据
    export_data = {
        "papers": papers,
        "edges": edges
    }

    # 导出为JSON
    with open('paper.json', 'w') as json_file:
        json.dump(export_data, json_file, ensure_ascii=False, indent=4)

finally:
    conn.close()

print("Export completed.")
