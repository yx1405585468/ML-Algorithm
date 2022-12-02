import yaml
import chinese_calendar as calendar

from atlassian import Jira
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# 获取 Jira 相关配置
def get_jira_config():
    with open('config.yml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config['Jira']


# 获取 Jira 对象
def get_jira(config):
    return Jira(url=config['url'], username=config['username'], password=config['password'])


# 组装 Jira SQL 语句
def _get_jql(search):
    # 按 jql 查询
    jql = search.pop('jql', None)
    if jql:
        return jql
    # 排序
    order_by = []
    orders = search.pop('order', None)
    if orders:
        if isinstance(orders, str):
            orders = [orders]
        for order in orders:
            if order.startswith('-'):
                order_by.append(f'{order[1:]} desc')
            else:
                order_by.append(f'{order} asc')
    # 其它查询条件
    content = []
    for key, value in search.items():
        if value:
            if isinstance(value, list):
                content.append(f'{key} in ({str(value)[1:-1]})')
            else:
                content.append(f'{key} = {value}')
    return ' and '.join(content) + ' order by ' + ', '.join(order_by) if order_by else ' and '.join(content)


# 获取 Jira 上的问题
def get_jira_issues(jira, config):
    jql = _get_jql(config['search'])
    result = jira.jql(jql, limit=1000)  # 限制最多拿1000条
    for issue in result['issues']:
        issue['url'] = f'{config["url"]}browse/{issue["key"]}'
        print(f'{issue["id"]}\t\t{issue["key"]}\t\t{issue["url"]}\t\t{issue["fields"]["summary"]}')
    return result['issues']


# 获取登录 Jira 的用户的 key 值
def _get_jira_user_key(jira, config):
    return jira.user(config['username'])['key']


# 获取指定 issue 的 id
def _get_jira_issue_id(jira, config):
    issue_key = config['tempo']['issue_key']
    if issue_key:
        return jira.jql(f'key = {issue_key}')['issues'][0]['id']
    else:
        return get_jira_issues(jira, config)[0]['id']


# 获取填工时的日期范围
def _get_date_range(config):
    if config['tempo']['start_date'] and config['tempo']['end_date']:
        start_date = datetime.strptime(config['tempo']['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(config['tempo']['end_date'], '%Y-%m-%d').date()
    else:  # 配置文件中没有开始日期和结束日期，默认为当月的开始日期和结束日期
        start_date = datetime.now().date().replace(day=1)
        end_date = start_date + relativedelta(months=1) - timedelta(days=1)
    return start_date, end_date


# 获取 Jira 上的工时
def get_jira_worklogs(jira, config):
    start_date, end_date = _get_date_range(config)
    worklogs = jira.tempo_timesheets_get_worklogs(date_from=start_date, date_to=end_date)
    for worklog in worklogs:
        print(
            f'{worklog["author"]["displayName"]}在{worklog["dateStarted"][:10]}日花费{worklog["timeSpentSeconds"] / 60 / 60}小时处理了{worklog["issue"]["key"]}({worklog["issue"]["summary"]})')
    return worklogs


# 获取已经填了工时的日期
def _get_worklog_dates(jira, config):
    worklogs = get_jira_worklogs(jira, config)
    return [worklog["dateStarted"][:10] for worklog in worklogs]


# 设置 Jira 上的工时
def set_jira_worklogs(jira, config):
    worker = _get_jira_user_key(jira, config)
    issue_id = _get_jira_issue_id(jira, config)
    start_date, end_date = _get_date_range(config)
    print(worker, issue_id, start_date, end_date)
    done_dates = _get_worklog_dates(jira, config)
    while start_date <= end_date:
        on_holiday, holiday_name = calendar.get_holiday_detail(start_date)
        if on_holiday:
            print(f'{start_date}{" 放 " + holiday_name + " 假期" if holiday_name else " 放周末"}，啦啦啦~~~~~~~~~~')
        else:
            if str(start_date) in done_dates:
                print(f'{start_date}日的工时已经填了')
            else:
                print(f'{start_date} 不放假，填工时！！！！！！！！！！')
                tempo_timesheets_write_worklog(jira, worker, str(start_date), 60 * 60 * 8, issue_id)
        start_date += timedelta(days=1)


# 原来的方法有问题，缺少 remainingEstimate 字段，故重写该方法
def tempo_timesheets_write_worklog(self, worker, started, time_spend_in_seconds, issue_id, comment=None):
    """
    Log work for user
    :param worker:
    :param started:
    :param time_spend_in_seconds:
    :param issue_id:
    :param comment:
    :return:
    """
    data = {
        "worker": worker,
        "started": started,
        "timeSpentSeconds": time_spend_in_seconds,
        "originTaskId": str(issue_id),
        "remainingEstimate": 0  # 缺少该字段
    }
    if comment:
        data["comment"] = comment
    url = "rest/tempo-timesheets/4/worklogs/"
    return self.post(url, data=data)


if __name__ == '__main__':
    config = get_jira_config()
    jira = get_jira(config)
    set_jira_worklogs(jira, config)
