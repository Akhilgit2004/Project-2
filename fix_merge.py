import json

with open('sentiment_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == 'merge':
        source = cell['source']
        new_source = []
        for line in source:
            if line == 'def merge_by_decade(text_lang_df, hist_df, language):\n':
                new_source.append('def merge_by_decade(text_lang_df, hist_df, language):\n')
                new_source.append('    hist_df = hist_df.copy()\n')
                new_source.append("    hist_df['decade'] = hist_df['decade'].astype(int)\n")
            else:
                new_source.append(line)
        cell['source'] = new_source
        break

with open('sentiment_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Done')