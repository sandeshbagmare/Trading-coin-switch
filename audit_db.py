import chromadb
client = chromadb.PersistentClient(path='data/knowledge/chromadb')
col = client.get_collection('trading_strategies')
print(f'Total chunks in DB: {col.count()}')
results = col.get(limit=3, include=['documents','metadatas'])
for d, m in zip(results['documents'], results['metadatas']):
    print(f"  [{m['source']} p{m['page']}] {d[:100]}...")
