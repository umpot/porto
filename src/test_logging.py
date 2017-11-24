import logging

logging.basicConfig(filename='log.log',level=logging.DEBUG, format='%(message)s')

def log(msg, *args):
    if len(args)>0:
        msg = msg.format(*args)
    print msg
    logging.info(msg)

log('test_{} {}', 'a', 12)