#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : Ryan Fan 
@E-Mail  : ryanfan0528@gmail.com
@Version : v1.0
"""
import os
import time
import sys
import re
import datetime
import pandas as pd 
import numpy as np
import warnings
import smtplib
from email.mime.text import MIMEText


class send_email(object):
    def __init__(self, receivers, final_top_frame, subject, table_name, time_line):
        self.receivers = receivers
        self.final_top_frame = final_top_frame
        self.subject = subject
        self.table_name = table_name
        self.date = time_line

    def generate_html_table(self):
        table_content = '\n<table style="empty-cells:show;border-collapse:collapse;border-spacing:0;font:18px Calibri;">\n'

        table_content += '<thead>\n'
        table_content += '\t<tr>\n'
        for column_name in list(self.final_top_frame):
            table_content += '\t\t<th style="border:1px solid #DDD;background:#f5f5f5;color:#333;line-border:1px solid #DDD;padding:5px 10px;background:#f5f5f5;color:#333;height:30px;text-align:left;word-break:break-all;font-size:15px;"><span style="width:200px">' + column_name + '</span></th>\n'
        table_content += '\t</tr>\n'
        table_content += '</thead>\n'

        table_content += '<tbody>\n'
        for row in self.final_top_frame.values:
            table_content += '\t<tr>\n'

            for column in row:
                if type(column) == int or type(column) == float:
                    column = `column`
                table_content += '\t\t<td style="padding:5px 10px;font-size:14px;width:150px;border-bottom:1px solid #DDD;border-bottom:1px solid #DDD;text-align:left;word-break:break-all;">' + column + '</td>\n'

            table_content += '\t</tr>\n'

        table_content += '</tbody>\n'
        table_content += '</table>'

        return table_content

    def generate_mail_context(self):
        context = '<B><font face="verdana" size="4" color="orange">%s:</font></B>'%(self.table_name)

        context += '<br>'
        context += self.generate_html_table()

        return context

    def run(self):
        # Define SMTP email server details
        smtp_server = 'smtp.qq.com'
        smtp_port = 25
        sender_username = 'fanpeng0313@vip.qq.com'
        sender_password = 'ocfdswzravahbjhd'

        # 生成邮件内容的html
        context = self.generate_mail_context()

        print "Send Mail From",sender_username
        print "Send Mail To", self.receivers

        msg = MIMEText(context, "html")
        msg['From'] = 'fanpeng0313@vip.qq.com'
        msg['To'] = ','.join(self.receivers)
        msg['Subject'] = self.subject + '  ' + self.date

        # Send email
        s = smtplib.SMTP(smtp_server, smtp_port)
        s.login(sender_username, sender_password)
        s.sendmail(msg['From'], self.receivers, msg.as_string())
        s.quit()

        print "Send E-Mail Successfully!"

    def email_error(self):
        # Define SMTP email server details
        smtp_server = 'smtp.qq.com'
        smtp_port = 25
        sender_username = 'fanpeng0313@vip.qq.com'
        sender_password = 'ocfdswzravahbjhd'

        # 发送的内容是错误代码
        error_msg = self.final_top_frame

        print "Send Mail From",sender_username
        print "Send Mail To", self.receivers

        msg = MIMEText(error_msg, 'plain', 'utf-8')
        msg['From'] = 'fanpeng0313@vip.qq.com'
        msg['To'] = ','.join(self.receivers)
        msg['Subject'] = self.subject + '  ' + self.date

        # Send email
        s = smtplib.SMTP(smtp_server, smtp_port)
        s.login(sender_username, sender_password)
        s.sendmail(msg['From'], self.receivers, msg.as_string())
        s.quit()

        print "Send E-Mail Successfully!"


# def main():
#     # 设置参数
#     receivers_1 = ['ryanfan0313@163.com']
#     Subject_1 = 'Email_Test'
#     table_name_1 = 'Today is a nice day'
#     date = '2018-07-25'
#     all_final_top_1 = pd.DataFrame({'ryan':[1,2,3],
#                                     'fan':[3,4,5]})

#     for column in all_final_top_1:
#         all_final_top_1[column] = all_final_top_1[column].astype(str)

#     send_email_func = send_email(receivers_1, all_final_top_1, Subject_1, table_name_1, date)
#     send_email_func.run()

# if __name__ == '__main__':
#     main()

