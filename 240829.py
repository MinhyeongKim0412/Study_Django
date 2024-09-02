#%%
import pandas as pd
#%%
data = pd.read_csv("C:/www2/iris/files/iris.csv")[:5]
#%%
for dt in  data.to_dict('records'):
    for k,v in dt.items():
        print(v)
    print("프린트 샘플")
# %%
dt.to_dict("records")
# %%
