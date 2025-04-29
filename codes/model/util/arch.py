def share_parameters(src, dst, shared_keywords=[]):
    dst_train_params = []
    src_params = {}
    for name, param in src.named_parameters():
        src_params[name] = param
    for name, param in dst.named_parameters():
        for shared_keyword in shared_keywords:
            if shared_keyword in name:
                param.data = src_params[name].clone()
                param.requires_grad = False
                print(f"name {name} are shared")
            else:
                dst_train_params.append(param)
    return dst_train_params


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner
