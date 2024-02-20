import subprocess
import time


def run_script():
    while True:
        try:
            # 运行脚本
            process = subprocess.Popen(['python', 'chda.py'])
            process.wait()  # 等待脚本运行完成

            print('process.returncode: ')
            print(process.returncode)
            # 如果脚本退出码为0，则正常结束循环
            if process.returncode == 0:
                break

            print('Script crashed. Restarting in 2 minutes...')
            time.sleep(120)  # 等待2分钟
        except Exception as e:
            print(f'Error occurred: {str(e)}')
            print('Restarting in 2 minutes...')
            time.sleep(120)  # 等待2分钟


run_script()