
def construct_ensemble_inp(config=None, **kwargs):
    resp_keys = list(kwargs.keys())
    for key in resp_keys:
        if key.startswith("merge_models_output"):
            del kwargs[key]
            break
    return kwargs
