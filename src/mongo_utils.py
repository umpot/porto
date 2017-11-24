from pymongo import MongoClient
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

gc_host='35.203.180.79'
local_host = '10.20.0.144'
user='ubik'
password='nfrf[eqyz'#nfrf[eqyz
db_name = 'porto'
collection_name='test1'

host = gc_host

client = MongoClient(host, 27017)
client['admin'].authenticate(user, password)
db = client[db_name]

# def load_results(name):
#     collection = db[name]
#     return [x['results'] for x in collection.find()]

# def load_importance(name, features):
#     arr = load_importance_raw(name)
#     sz = len(features)
#     res = [np.mean([x[j] for x in arr]) for j in range(sz)]
#     stds = [np.std([x[j] for x in arr]) for j in range(sz)]
#     res = zip(features, res, stds)
#     res.sort(key=lambda s: s[1], reverse=True)
#     return res



def load_importance_raw(name):
    collection = db[name]
    return [[y['importance'] for y in x['folds_info']] for x in collection.find()]

def cv_importance_to_df(imp):
    names = []
    for i in imp:
        names+=list(i.keys())
    names = list(set(names))
    vals=[[x.get(n,0) for x in imp] for n in names]
    df = pd.DataFrame({'name':names, 'importance':vals})
    df['mean'] = df['importance'].apply(np.mean)
    return df[['name','mean' ,'importance']].sort_values('mean', ascending=False)

# def explore_importance(name, features, N=None):
#     if N is None:
#         N=len(features)
#
#     res = load_importance(name, features)
#     print res
#     res=res[:N]
#     xs = [x[0] for x in res]
#     ys=[x[1] for x in res]
#     sns.barplot(xs, ys)
#     sns.plt.show()

def plot_lambda():
    name = 'porto_optimizing_lambda_v1'
    collection = db[name]
    res = [(x['params']['lambda'], x['avg_performance'], x['performance']) for x in collection.find()]
    first = res[::2]
    second = res[1::2]
    plt.plot([x[0] for x in first], [x[1] for x in first])
    plt.plot([x[0] for x in second], [x[1] for x in second])
    plt.show()

def plot_lambda_detail_folds():
    name = 'porto_optimizing_lambda_v1'
    collection = db[name]
    res = [(x['params']['lambda'], x['avg_performance'], x['performance']) for x in collection.find()]
    first = res[::2]
    second = res[1::2]

    for i in range(5):
        plt.plot([x[0] for x in first], [x[2][i] for x in first])

    for i in range(5):
        plt.plot([x[0] for x in second], [x[2][i] for x in second])

    plt.show()



