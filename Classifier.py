from Actors import start_classify_nanodet, start_classify_yolo8


class Classifier:

    def __init__(self):
        self.__classifierType = "Yolo8"  # По умолчанию модель поумнее

    def set_classifier_type(self, type):
        if type == "Yolo8" or type == "NanoDet":
            self.__classifierType = type
        else:
            print(f"Модели {type} нет в боте")

    def get_classifier_type(self):
        return self.__classifierType

    def start_classify(self, file_name, user_id, message_json: str):
        if self.get_classifier_type() == "Yolo8":
            start_classify_yolo8.send(file_name, user_id, message_json)
        if self.get_classifier_type() == "NanoDet":
            start_classify_nanodet.send(file_name, user_id, message_json)
