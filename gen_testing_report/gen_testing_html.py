import yaml

from jira_utils import get_jira_config, get_jira, get_jira_issues


def get_ikas_config():
    with open('config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config['IKAS']


# 转测申请
def _gen_title(config):
    return f"""
    <tr height="50" style='height:25.00pt;mso-height-source:userset;mso-height-alt:500;'>
        <td height="50" style='height:25.00pt;'></td>
        <td class="xl65" colspan="4" style='border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;' x:str>{config['title']}</td>
    </tr>
    """


# 项目迭代版本
def _gen_version(config):
    return f"""
    <tr height="50" style='height:25.00pt;mso-height-source:userset;mso-height-alt:500;'>
        <td height="50" style='height:25.00pt;'></td>
        <td class="xl66" x:str>项目迭代版本</td>
        <td class="xl67" colspan="3" style='border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;' x:str>{config['version']}</td>
    </tr>
    """


# 需求列表
def _gen_content(issues):
    content = f"""	
    <tr height="44" style='height:22.00pt;mso-height-source:userset;mso-height-alt:440;'>
        <td height="44" style='height:22.00pt;'></td>
        <td class="xl66" rowspan="{len(issues) + 1}" style='border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;' x:str>需求列表</td>
        <td class="xl68" x:str>需求编号</td>
        <td class="xl68" x:str>需求描述</td>
        <td class="xl68" x:str>JIRA链接</td>
    </tr>
    """
    for issue in issues:
        content += f"""
            <tr height="36" style='height:18.00pt;mso-height-source:userset;mso-height-alt:360;'>
                <td height="36" style='height:18.00pt;'></td>
                <td class="xl69" x:str>{issue['key']}</td>
                <td class="xl69" x:str>{issue['fields']['summary']}</td>
                <td class="xl70" x:str>
                    <a href="{issue['url']}" target="_parent">{issue['url']}</a>
                </td>
            </tr>
        """
    return content


# 单元测试是否通过
def _gen_unit_testing(config):
    return f"""
    <tr height="44" style='height:22.00pt;mso-height-source:userset;mso-height-alt:440;'>
        <td height="44" style='height:22.00pt;'></td>
        <td class="xl71" x:str>单元测试是否通过</td>
        <td class="xl72" colspan="3" style='border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;' x:str>{config['pass_unit_testing']}</td>
    </tr>
    """


# 集成测试是否通过
def _gen_integration_testing(config):
    return f"""
    <tr height="44" style='height:22.00pt;mso-height-source:userset;mso-height-alt:440;'>
        <td height="44" style='height:22.00pt;'></td>
        <td class="xl71" x:str>集成测试是否通过</td>
        <td class="xl72" colspan="3" style='border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;' x:str>{config['pass_integration_testing']}</td>
    </tr>
    """


# 功能自测是否通过
def _gen_self_testing(config):
    return f"""
    <tr height="44" style='height:22.00pt;mso-height-source:userset;mso-height-alt:440;'>
        <td height="44" style='height:22.00pt;'></td>
        <td class="xl71" x:str>功能自测是否通过</td>
        <td class="xl72" colspan="3" style='border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;' x:str>{config['pass_self_testing']}</td>
    </tr>
    """


# 测试地址
def _gen_testing_url(config):
    return f"""
    <tr height="44" style='height:22.00pt;mso-height-source:userset;mso-height-alt:440;'>
        <td height="44" style='height:22.00pt;'></td>
        <td class="xl66" x:str>测试地址</td>
        <td class="xl75" colspan="3" style='border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;' x:str>
            <a href="{config['testing_url']}" target="_parent">{config['testing_url']}</a>
        </td>
    </tr>
    """


# 接口文档地址
def _gen_document_url(config):
    return f"""
    <tr height="44" style='height:22.00pt;mso-height-source:userset;mso-height-alt:440;'>
        <td height="44" style='height:22.00pt;'></td>
        <td class="xl66" x:str>接口文档地址:</td>
        <td class="xl78" colspan="3" style='border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;' x:str>
            <a href="{config['document_url']}" target="_parent">{config['document_url']}</a>
        </td>
    </tr>
    """


# jenkins构建地址
def _gen_jenkins_url(config):
    return f"""
    <tr height="44" style='height:22.00pt;mso-height-source:userset;mso-height-alt:440;'>
        <td height="44" style='height:22.00pt;'></td>
        <td class="xl66" x:str>jenkins构建地址</td>
        <td class="xl75" colspan="3" style='border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;' x:str>
            <a href="{config['jenkins_url']}" target="_parent">{config['jenkins_url']}</a>
        </td>
    </tr>
    """


# 项目总负责人
def _gen_project_leader(config):
    return f"""
    <tr height="44" style='height:22.00pt;mso-height-source:userset;mso-height-alt:440;'>
        <td height="44" style='height:22.00pt;'></td>
        <td class="xl66" x:str>项目总负责人</td>
        <td class="xl79" colspan="3" style='border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;'>{config['project_leader']}</td>
    </tr>
    """


# 项目成员
def _gen_project_members(config):
    project_members = config['project_members']
    return f"""
    <tr height="84" style='height:42.00pt;mso-height-source:userset;mso-height-alt:840;'>
        <td height="84" style='height:42.00pt;'></td>
        <td class="xl66" x:str>项目成员</td>
        <td class="xl79" colspan="3" style='border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;' x:str>【项目经理】：{project_members['project_manager']}<span style='mso-spacerun:yes;'>&nbsp;&nbsp;&nbsp; </span>【产品经理】：{project_members['product_manager']}<span style='mso-spacerun:yes;'>&nbsp; </span><br/>【前端】：{project_members['frontend']}<span style='mso-spacerun:yes;'>&nbsp; </span>【后台】：{project_members['backend']}<span style='mso-spacerun:yes;'>&nbsp; </span>【算法】：{project_members['algorithm']}<span style='mso-spacerun:yes;'>&nbsp; </span>【测试】：{project_members['testing']}</td>
    </tr>
    """


# 备注
def _gen_remark(config):
    return f"""
    <tr height="48" style='height:24.00pt;mso-height-source:userset;mso-height-alt:480;'>
        <td height="48" style='height:24.00pt;'></td>
        <td class="xl66" x:str>备注</td>
        <td class="xl80" colspan="3" style='border-right:1.0pt solid windowtext;border-bottom:1.0pt solid windowtext;'>{config['remark']}</td>
    </tr>
    """


# 生成html
def gen_html(output=True, output_filename='ikas_testing.html'):
    # 获取 jira 问题
    config = get_jira_config()
    jira = get_jira(config)
    issues = get_jira_issues(jira, config)
    # 拼接 html
    config = get_ikas_config()
    body = ''
    body += _gen_title(config)
    body += _gen_version(config)
    body += _gen_content(issues)
    body += _gen_unit_testing(config)
    body += _gen_integration_testing(config)
    body += _gen_self_testing(config)
    body += _gen_testing_url(config)
    body += _gen_document_url(config)
    body += _gen_jenkins_url(config)
    body += _gen_project_leader(config)
    body += _gen_project_members(config)
    body += _gen_remark(config)
    with open('template.html', 'r') as f:
        html = f.read().replace('<table-content>', body)
    if output:
        with open(output_filename, 'w') as f:
            f.write(html)
    return html


if __name__ == '__main__':
    html = gen_html()
