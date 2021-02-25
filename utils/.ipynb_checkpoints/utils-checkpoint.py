import pickle

def path(str_):
    return str_

def dump_pickle(file: str, object_):
    with open(path(file), "wb") as p_f:
        pickle.dump(object_, p_f)

def load_pickle(file: str):
    with open(path(file), "rb") as p_f:
        object_ = pickle.load(p_f)
        return object_