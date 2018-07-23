# coding: utf-8
__author__ = 'wangkongtao'
__version__ = '0.1'

from spyne.decorator import rpc
from spyne.service import ServiceBase
from spyne.model.complex import Iterable
from spyne.model.primitive import Unicode
from spyne.application import Application
from spyne.protocol.soap import Soap11
from spyne.server.wsgi import WsgiApplication
import undervoice_conv_muliti as uvm
import pymysql


class uvwebdService(ServiceBase):
    @rpc(_returns=Iterable(Unicode))
    def main_func(ctx):
        return_code = uvm.main(1)
        print return_code
        if return_code[0] == -1:
            yield str(-1)
            yield return_code[1]
            yield str('')
        else:
            yield str(0)
            yield return_code[1]
            conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice',
                                   charset='utf8')
            cursor = conn.cursor()
            sql_str = 'select distinct wav_index, bad_prob from runtime_prob where taskkey = "%s" and bad_flag = 1 order by bad_prob ' \
                      'DESC, wav_index LIMIT 10;' % return_code[1]
            print sql_str
            cursor.execute(sql_str)
            bad_wav_list = cursor.fetchall()
            print bad_wav_list
            yield ','.join(i[0] for i in bad_wav_list)


if __name__ == '__main__':
    application = Application([uvwebdService], 'spyne.hms.soap',
                              in_protocol=Soap11(validator='lxml'),
                              out_protocol=Soap11())
    wsgi_application = WsgiApplication(application)
    import logging
    from wsgiref.simple_server import make_server

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('spyne.protocol.xml').setLevel(logging.DEBUG)
    logging.info("listening to http://192.168.1.5:8619")
    logging.info("wsdl is at: http://192.168.1.5:8619/?wsdl")
    server = make_server('192.168.1.5', 8619, wsgi_application)
    server.serve_forever()
