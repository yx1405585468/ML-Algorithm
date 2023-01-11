import pandas as pd
import json

__all__ = ['add_stage_and_json_format']


def add_stage_and_json_format(stage: str) -> callable:
    # 根据stage 选择不同的验证方式, 并对函数的输入和输出json 格式化
    # todo: stage 验证

    def json_wrapper(func: callable) -> callable:
        # 定义一个闭包，实现函数的输入和输出 都是json

        def wrapper(json_data: 'json') -> 'json':
            # 导入json str
            json_input = json.loads(json_data)
            print(json_input.keys())
            print("--" * 100)
            print('访问 stage', stage)
            # 判断 data_resoure
            assert json_input['data_resource'] in ('training', 'testing', 'deploy'), ValueError(
                'Data resource is invalid!')

            if stage == 'read_data':
                pass
            elif stage != 'read_data':
                print(f'{stage} is here')
                json_input['data'] = pd.read_json(json_input['data'])

            result = func(**json_input)

            json_output = json.dumps(
                {
                    'data_resource': json_input['data_resource'],
                    'data': result.to_json()
                }
            )

            return json_output

        return wrapper

    return json_wrapper


@add_stage_and_json_format(stage='read_data')
def read_csv_file(filepath: str, data_resource: str) -> pd.DataFrame:
    # 读取数据文件
    return pd.read_csv(filepath)
