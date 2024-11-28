import dramatiq
import telebot
from dramatiq.brokers.rabbitmq import RabbitmqBroker
from telebot import types
import boto3

from Classifier import Classifier
from StartWorker import start_worker


def main():

    classifier = (
        Classifier()
    )  # Создаем экземпляр классификатора, с которым мбудем дальше работать

    bot = telebot.TeleBot(
        "7330326462:AAHquF2epd3qhhV6psyF2n2el4ez6P68HxA"
    )  # Подключаемся в боту по специальному ключу

    @bot.message_handler(commands=["start"])
    def start(message):

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton("❓Что ты умеешь❓")
        btn2 = types.KeyboardButton("Приступить к классификации")
        markup.add(btn1, btn2)
        bot.send_message(
            message.from_user.id,
            format(
                "Привет, я бот для классификации изображений\n↓↓↓Выберите желаемое действие↓↓↓"
            ),
            reply_markup=markup,
        )

    # Основной обработчик текстовых команд
    # Можно было бы создать много обработчиков и вместо кучи if использовать лямбды
    @bot.message_handler(content_types=["text"])
    def text_handler(message):

        if message.text == "❓Что ты умеешь❓":
            bot.send_message(
                message.from_user.id,
                format(
                    "Я бот для классификации разных изображений\nВ меня встроены две библиотеки: NanoDet и Yolo8\nК сожалению мои навыки весьма ограниченны, поэтому, пожалуйста, не ругайтесь на ошибки"
                ),
            )

        if message.text == "Приступить к классификации":
            markup = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
            btn1 = types.KeyboardButton(format("NanoDet"))
            btn2 = types.KeyboardButton(format("Yolo8"))
            back = types.KeyboardButton("Вернуться в главное меню")
            # markup.add(btn1, btn2, back)
            markup.add(btn1, btn2, back)
            bot.send_message(
                message.from_user.id,
                "Чем хотите классифицировать ?",
                reply_markup=markup,
            )

        if message.text == "NanoDet":
            classifier.set_classifier_type("NanoDet")
            bot.send_message(message.from_user.id, "Загрузите изображение")

        if message.text == "Yolo8":
            classifier.set_classifier_type("Yolo8")
            bot.send_message(message.from_user.id, "Загрузите изображение")

        if message.text == "Вернуться в главное меню":
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn1 = types.KeyboardButton("❓Что ты умеешь❓")
            btn2 = types.KeyboardButton("Приступить к классификации")
            markup.add(btn1, btn2)
            bot.send_message(
                message.from_user.id, "Выберите действие", reply_markup=markup
            )

    @bot.message_handler(content_types=["photo"])
    def photo_handler(message: telebot.types.Message):

        # Создаем подключение к облачному хранилищу
        s3 = boto3.client(
            "s3",
            endpoint_url="https://s3.storage.selcloud.ru",
            region_name="ru-1a",
            aws_access_key_id="f63103862e52488cbfa88e2cff776669",
            aws_secret_access_key="c65fdffbd4bc4fe1bbd93ea164cbe2c6",
        )
        bucket_name = "zvr-bot-storage"

        # Загружаем фото из ТГ
        fileID = message.photo[-1].file_id
        file_info = bot.get_file(fileID)
        downloaded_file = bot.download_file(file_info.file_path)
        file_name = file_info.file_path.split("/")[-1]

        # Отпарвляем в облако и на классификацию
        photo = open(file_name, "wb")
        photo.write(downloaded_file)
        photo.close()  # Закрываем ресурс
        user_id = message.from_user.id
        message_json = message.json
        s3.upload_file(file_name, bucket_name, file_name)
        classifier.start_classify(file_name, user_id, message_json)
        bot.send_message(
            chat_id=message.from_user.id,
            text=f"Ваш запрос обрабатывается моделью {classifier.get_classifier_type()}\nМожете сменить модель или загрузить следующее изображение",
            reply_to_message_id=message.message_id,
        )

    # Бесконечное прослушивание бота
    bot.infinity_polling()


if __name__ == "__main__":

    rabbitmq_broker = RabbitmqBroker(host="rabbitmq")
    dramatiq.set_broker(rabbitmq_broker)
    start_worker()
    main()
