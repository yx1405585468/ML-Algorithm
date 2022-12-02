import smtplib
import yaml

from email.header import Header
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from gen_testing_html import gen_html


def get_email_config():
    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config['Email']


def send_mail(content, server, auth, subject, f, to, cc, attachments):
    message = MIMEMultipart()
    # 配置邮件主题
    message['Subject'] = Header(subject, 'utf-8')
    # 发件邮箱
    message['From'] = f
    # 配置收件人
    message['To'] = ";".join(to)
    # 配置抄送人
    message['Cc'] = ";".join(cc)
    message.attach(MIMEText(content, _subtype="html", _charset='utf-8'))
    # 添加附件
    for attachment in attachments:
        try:
            file = MIMEApplication(open(attachment.encode('utf-8'), 'rb').read())
            file.add_header('Content-Disposition', 'attachment', filename=f'{attachment.split("/")[-1]}')
            message.attach(file)
        except FileNotFoundError:
            raise ValueError(f"文件 {attachment} 不存在")
    try:
        smtp_server = smtplib.SMTP_SSL(*server)
        smtp_server.login(*auth)
        smtp_server.sendmail(f, to + cc, message.as_string())
        smtp_server.quit()
        print("邮件发送成功.")
    except smtplib.SMTPException as e:
        raise ValueError(f"邮件发送失败, {e}")


if __name__ == "__main__":
    config = get_email_config()
    html = gen_html()
    send_mail(content=html, **config)
