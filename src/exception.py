
import sys
import logging
from src.logger import logging

def error_message_detai1(error, error_detail: sys):
    _, _, tb = sys.exc_info()

    file_name = tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, tb.tb_lineno, str(error))
    return error_message


class CustomException(Exception):

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detai1(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message


if __name__ == "__main__":
    try:
        a= 1/0
    except Exception as e:
        logging.info("Devide by zero error")
        raise CustomException(e, sys)
