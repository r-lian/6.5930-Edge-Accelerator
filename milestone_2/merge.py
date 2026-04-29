import json, sys                                                  
a = json.load(open('cache_exp_1.json'))                                 
b = json.load(open('cache_exp_2.json'))                                                                                                               
merged = {**a, **b}                                     
print(f'a: {len(a)} keys, b: {len(b)} keys, merged: {len(merged)} keys (overlap: {len(a)+len(b)-len(merged)})', file=sys.stderr)                  
json.dump(merged, open('merged_cache.json', 'w'), indent=2)
