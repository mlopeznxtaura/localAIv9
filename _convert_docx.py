#!/usr/bin/env python3
import zipfile, json, re, sys

docx_path = "1stdataset512.json.docx"
json_path = "output/1stdataset512.json"

try:
    with zipfile.ZipFile(docx_path, 'r') as z:
        content = z.read('word/document.xml').decode('utf-8')
        text = re.sub(r'<[^>]+>', '', content).strip()
        
        objects = []
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    obj_str = text[start:i+1]
                    try:
                        obj = json.loads(obj_str)
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None
        
        if objects:
            with open(json_path, 'w') as f:
                json.dump(objects, f, indent=2)
            print(f"Converted {len(objects)} objects -> {json_path}")
        else:
            print("No valid JSON objects found")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
