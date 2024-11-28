import os
import boto3
import dramatiq
import telebot
from telebot.types import Message
from NanodetRun import classify_nanodet
from Yolo8 import classify_yolo8


@dramatiq.actor(queue_name="Yolo8")
def start_classify_yolo8(file_name, user_id, message_json: str):

    bot = telebot.TeleBot("7330326462:AAHquF2epd3qhhV6psyF2n2el4ez6P68HxA")

    s3 = boto3.client(
        "s3",
        endpoint_url="https://s3.storage.selcloud.ru",
        region_name="ru-1a",
        aws_access_key_id="f63103862e52488cbfa88e2cff776669",
        aws_secret_access_key="c65fdffbd4bc4fe1bbd93ea164cbe2c6",
    )
    bucket_name = "zvr-bot-storage"

    # Дополнительная загрузка файла из облака нужна на случай, если пользователь пришлёт одну и ту же фотографию
    # (ТГ присваивает им одинаковые имена) и один из потоков её удалит, хотя она понадобится другому
    s3.download_file(bucket_name, file_name, file_name)

    time = classify_yolo8(file_name)
    reply_mes_name = file_name.split(".")[0] + "_yolo8.jpg"
    reply_photo = open(reply_mes_name, "rb")
    bot.send_photo(
        user_id,
        reply_photo,
        reply_to_message_id=Message.de_json(message_json).message_id,
        caption=time + "\nРезультат классификации Yolo8:",
    )
    reply_photo.close()
    try:
        os.remove(reply_mes_name)
        os.remove(file_name)
    except:
        pass


@dramatiq.actor(queue_name="NanoDet")
def start_classify_nanodet(file_name, user_id, message_json: str):

    bot = telebot.TeleBot("7330326462:AAHquF2epd3qhhV6psyF2n2el4ez6P68HxA")

    s3 = boto3.client(
        "s3",
        endpoint_url="https://s3.storage.selcloud.ru",
        region_name="ru-1a",
        aws_access_key_id="f63103862e52488cbfa88e2cff776669",
        aws_secret_access_key="c65fdffbd4bc4fe1bbd93ea164cbe2c6",
    )
    bucket_name = "zvr-bot-storage"

    s3.download_file(bucket_name, file_name, file_name)

    time = classify_nanodet(file_name)
    reply_mes_name = file_name.split(".")[0] + "_NanoDet." + file_name.split(".")[1]
    reply_photo = open(reply_mes_name, "rb")
    bot.send_photo(
        user_id,
        reply_photo,
        reply_to_message_id=Message.de_json(message_json).message_id,
        caption=time + "\nРезультат классификации NanoDet:",
    )
    reply_photo.close()
    try:
        os.remove(reply_mes_name)
        os.remove(file_name)
    except:
        pass
