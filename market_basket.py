# ============================================
# Market Basket Analysis
# Codec Technologies - Industrial Project
# ============================================

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("   MARKET BASKET ANALYSIS PROJECT")
print("=" * 50)

# ── 1. LOAD DATA ──────────────────────────────
print("\n📂 Loading transactions...")
transactions = []
with open('data/groceries.csv', 'r') as f:
    for line in f:
        items = line.strip().split(',')
        transactions.append(items)
print(f"✅ Total transactions: {len(transactions)}")

# ── 2. ENCODE DATA ────────────────────────────
print("\n⚙️  Encoding transactions...")
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(f"✅ Items found: {len(te.columns_)}")

# ── 3. APRIORI ALGORITHM ──────────────────────
print("\n🔍 Running Apriori algorithm...")
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
print(f"✅ Frequent itemsets found: {len(frequent_itemsets)}")

# ── 4. ASSOCIATION RULES ──────────────────────
print("\n📋 Generating association rules...")
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=len(frequent_itemsets))
rules = rules.sort_values('lift', ascending=False)
print(f"✅ Rules generated: {len(rules)}")

# ── 5. TOP RULES ──────────────────────────────
print("\n🏆 Top 10 Association Rules:")
print("=" * 60)
top_rules = rules.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
for _, row in top_rules.iterrows():
    ant = ', '.join(list(row['antecedents']))
    con = ', '.join(list(row['consequents']))
    print(f"  {ant} → {con}")
    print(f"  Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.4f}")
    print()

# ── 6. VISUALIZATIONS ─────────────────────────
print("📊 Creating visualizations...")

# Top 10 most frequent items
item_counts = df.sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
bars = plt.bar(item_counts.index, item_counts.values, color=plt.cm.Set2.colors[:10])
plt.title('Top 10 Most Frequent Items', fontsize=16, fontweight='bold')
plt.xlabel('Items')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_items.png', dpi=150)
print("✅ top_items.png saved!")

# Support vs Confidence scatter
plt.figure(figsize=(10, 6))
scatter = plt.scatter(rules['support'], rules['confidence'],
                      c=rules['lift'], cmap='YlOrRd', alpha=0.6, s=50)
plt.colorbar(scatter, label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence (colored by Lift)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('support_confidence.png', dpi=150)
print("✅ support_confidence.png saved!")

# Heatmap of top rules
top20 = rules.head(20).copy()
top20['rule'] = top20.apply(lambda x: f"{', '.join(list(x['antecedents']))} → {', '.join(list(x['consequents']))}", axis=1)
plt.figure(figsize=(10, 8))
heatmap_data = top20[['support', 'confidence', 'lift']].set_index(top20['rule'])
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='Blues', linewidths=0.5)
plt.title('Top 20 Rules Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rules_heatmap.png', dpi=150)
print("✅ rules_heatmap.png saved!")

print("\n" + "=" * 50)
print("   ✅ PROJECT COMPLETE!")
print("=" * 50)
print(f"\n📊 Summary:")
print(f"   Total Transactions : 9,835")
print(f"   Unique Items       : {len(te.columns_)}")
print(f"   Frequent Itemsets  : {len(frequent_itemsets)}")
print(f"   Association Rules  : {len(rules)}")