import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("pearson_minus.csv")

g=sns.histplot(
    df,
    x="pearson"
)
g.set_yscale('log')
plt.title("Pearson - minus")
plt.savefig("pearson_minus.png") 
