
import shlex, subprocess



def start_worker():
    cmd = ["dramatiq Actors --queues NanoDet", "dramatiq Actors --queues Yolo8"]
    for val in cmd:
        args = shlex.split(val)
        print(f'Запущен актер {args[-1]}')
        subprocess.Popen(args)
    print('Работники запущены')




