from coding.view import app, db
import time
import logging

if __name__ == "__main__":
    # handler = logging.FileHandler('flask.log')
    # app.logger.addHandler(handler)
    logger = logging.getLogger('/var/log/nlp/nlp_classes.log')
    logger.setLevel(logging.DEBUG)
    formatter = '%(asctime)s: %(levelname)s %(filename)s-%(module)s-%(funcName)s-%(lineno)d %(message)s'
    log_formatter = logging.Formatter(formatter)
    handler = logging.FileHandler('/var/log/nlp/nlp_classes.log')
    logger.addHandler(handler)
    app.run(port=8888)
