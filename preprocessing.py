import pandas as pd

# 1. Φόρτωση ΜΟΝΟ του train set
train_df = pd.read_csv('/Users/antonistsipoulakos/Downloads/UNSW_NB15_training-set(in).csv')

print("Το train set φορτώθηκε! Αρχικό μέγεθος:", train_df.shape)

# 2. Διαγραφή των στηλών 'id' και 'attack_cat'
# axis=1 σημαίνει ότι διαγράφουμε στήλες 
train_df = train_df.drop(['id', 'attack_cat'], axis=1)

print("Οι στήλες διαγράφηκαν! Νέο μέγεθος:", train_df.shape)

# 3. Εμφάνιση των πρώτων 5 γραμμών για οπτικό έλεγχο
print(train_df.head())





#ΕΠΟΜΕΝΕΣ ΑΛΛΑΓΕΣ

# 1. Διαχωρισμός των χαρακτηριστικών (X) από τον στόχο (y)
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

print("Ο διαχωρισμός σε X_train και y_train έγινε επιτυχώς!")

# 2. Έλεγχος για κενές τιμές (Βήμα 'α' της εκφώνησης)
# Το συγκεκριμένο αρχείο συνήθως είναι καθαρό, αλλά καλό είναι να το ελέγχουμε
missing_values = X_train.isnull().sum().sum()
print(f"Συνολικές κενές τιμές στο X_train: {missing_values}")

# Εάν missing_values > 0, θα χρειαζόταν συμπλήρωση (π.χ. με X_train = X_train.fillna(0)). 
# Λογικά θα σου βγάλει 0.

# 3. Μετατροπή ποιοτικών χαρακτηριστικών σε 0/1 (Dummy Variables - Βήμα 'β')
categorical_cols = ['proto', 'service', 'state']

# Η εντολή pd.get_dummies φτιάχνει αυτόματα τις νέες στήλες με τα 0/1
X_train = pd.get_dummies(X_train, columns=categorical_cols)

print("Οι ποιοτικές μεταβλητές μετατράπηκαν σε Dummies!")
print("Νέο μέγεθος X_train (παρατηρούμε ότι οι στήλες αυξήθηκαν σημαντικά):", X_train.shape)