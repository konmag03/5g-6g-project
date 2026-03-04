import pandas as pd

# 1. Φόρτωση του train set (EXCEL, όχι CSV)
train_df = pd.read_excel(r"C:\Users\ldako\Downloads\UNSW_NB15_training-set(in).xlsx")

print("Το train set φορτώθηκε! Αρχικό μέγεθος:", train_df.shape)

# 2. Διαγραφή των στηλών 'id' και 'attack_cat'
train_df = train_df.drop(['id', 'attack_cat'], axis=1)

print("Οι στήλες διαγράφηκαν! Νέο μέγεθος:", train_df.shape)

# 3. Εμφάνιση των πρώτων 5 γραμμών
print(train_df.head())
