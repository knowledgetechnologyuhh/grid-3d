import torch.nn as nn


def unflatten_json(json):
    if type(json) == dict:
        for k in sorted(json.keys(), reverse=True):
            if "." in k:
                key_parts = k.split(".")
                json1 = json
                for i in range(0, len(key_parts) - 1):
                    k1 = key_parts[i]
                    if k1 in json1:
                        json1 = json1[k1]
                        if type(json1) != dict:
                            conflicting_key = ".".join(key_parts[0 : i + 1])
                            raise Exception(
                                'Key "{}" conflicts with key "{}"'.format(
                                    k, conflicting_key
                                )
                            )
                    else:
                        json2 = dict()
                        json1[k1] = json2
                        json1 = json2
                if type(json1) == dict:
                    v = json.pop(k)
                    json1[key_parts[-1]] = v


def gen_feature_extractor(model, output_layer=None):
    layers = list(model._modules.keys())
    layer_count = 0
    for layer in layers:
        if layer != output_layer:
            layer_count += 1
        else:
            break
    for i in range(1, len(layers) - layer_count):
        model._modules.pop(layers[-i])
    feature_extractor = nn.Sequential(model._modules)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval()
    return feature_extractor
