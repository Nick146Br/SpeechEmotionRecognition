import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas

current_dir = os.getcwd().replace('\\', '/')
models = ['GCGRU']
tasks = ['BBP']

color1 = sns.color_palette('tab10')
color2 = sns.color_palette('pastel')
color3 = sns.color_palette('bright')

for task in tasks:
    index = []
    dict_models = {}
    for type_model in models:
        acc = []
        path = os.path.join(current_dir, "Logs", "Results", type_model)
        files = sorted(os.listdir(path))
        files = [fname for fname in files if fname.endswith('.csv')]
        last_file = files[-1]
        path = os.path.join(path, last_file)
        df = pandas.read_csv(path)
        for idx in df.index:
            if(len(index) < len(df.index)): index.append(df['Subject ID'][idx])
            acc.append(df['Accuracy'][idx])
        
        while(len(acc) < len(index)): 
            acc.append(0.0)
        dict_models[type_model] = acc

    df_new = pandas.DataFrame(dict_models, index=index)
    os.makedirs(os.path.dirname(os.path.join(current_dir, "Logs", "Imgs", type_model, task + "_" + type_model + "_bar.png")), exist_ok=True)
    df_new.plot(kind='bar', color=[color1[9]])
    plt.xlabel('Subject ID')
    plt.ylabel('Accuracy')
    plt.title(task + " - " + type_model)
    plt.ylim([0, 1.2])
    plt.legend(loc='upper center', ncol=4)
    plt.savefig(os.path.join(current_dir, "Logs", "Imgs", type_model, task + "_" + type_model + "_bar.png"))
    # plt.show()
    plt.close()
        
        