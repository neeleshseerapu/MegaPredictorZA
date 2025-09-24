import pandas as pd

df = pd.read_csv('Pokedex_Ver_SV2.csv')

df.drop(columns=['E_HP', 'E_Attack', 'E_Defense', 'E_SP_Attack', 'E_SP_Defense',
       'E_Speed', 'Color'], inplace=True)

megadf = df[df["Mega_Evolution_Flag"]=="Mega"].drop_duplicates(subset="No")
megadf = megadf.set_index("No")

regiondf = df[df["Region_Form"].notna()].drop_duplicates(subset="No")
regiondf = regiondf.set_index("No")

df.drop(columns=['Mega_Evolution_Flag', 'Region_Form', 'Branch_Code'], inplace=True)
df.drop_duplicates(subset=['No'], inplace=True)
df.reset_index(drop=True, inplace=True)

df["Regional"] = 0
df.loc[regiondf.index-1, "Regional"] = 1

df["Mega"] = 0
df.loc[megadf.index-1, "Mega"] = 1

df = df.set_index("No")

df.to_csv("cleandex.csv")

print(df.head(26))

