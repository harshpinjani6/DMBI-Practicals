"""DMBI Experiment 9: Apriori association rule mining on Groceries dataset."""
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Keep Groceries_dataset.csv in the same folder as this script.
df = pd.read_csv('Groceries_dataset.csv')
print('Dataset loaded successfully!')
print(df.head())
print('Dataset shape:', df.shape)

transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).tolist()
print('Total transactions:', len(transactions))
print('First five transactions:')
for transaction in transactions[:5]:
    print(transaction)

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
encoded_df = pd.DataFrame(te_array, columns=te.columns_)
print('Transaction encoding completed!')
print(encoded_df.head())
print('Transaction matrix shape:', encoded_df.shape)

# Thresholds match the code/screenshots in the submitted DOCX.
frequent_itemsets = apriori(encoded_df, min_support=0.001, use_colnames=True)
print('Frequent Itemsets:')
print(frequent_itemsets.head(10))
print('Total frequent itemsets:', len(frequent_itemsets))

rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.1)
rules = rules.sort_values(by='lift', ascending=False)
print('Total association rules:', len(rules))
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

if not rules.empty:
    print('Top Association Rule:')
    print(rules.iloc[0][['antecedents', 'consequents', 'support', 'confidence', 'lift']])
else:
    print('No association rules found. Try adjusting the thresholds.')
