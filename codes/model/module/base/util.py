def extend2tuple(param, num, check=True, dim=-1):
    if dim < 2:
        if isinstance(param, (list, tuple)):
            if len(param) == num:
                param_list = list(param)
            else:
                assert len(param) > num
                param_list = list(param)[:num]
        else:
            param_list = [param] * num
    else:
        if isinstance(param, (list, tuple)):
            if check:
                if len(param) == num:
                    param_list = list(param)
                elif len(param) == dim:
                    param_list = [param] * num
                else:
                    raise ValueError(f"param {param} of multiple layers is not supported!")
            else:
                assert len(param) > num
                param_list = list(param)[:num]
        else:
            param_list = [param] * num
    return tuple(param_list)
