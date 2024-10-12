import yaml

def get_model_params(args):
    model_name=args.model_name
    yaml_file=f'./Yaml/{model_name}.yaml'
    with open(yaml_file, 'r') as f:
        hparam_dict = yaml.load(f, yaml.FullLoader)
    # 更新 args 对象
    for key, value in hparam_dict.items():
        setattr(args, key, value)
    return args # 返回更新后的args对象
