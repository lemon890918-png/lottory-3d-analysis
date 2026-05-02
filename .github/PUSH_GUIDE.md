# GitHub Push Guide (lottory-3d-analysis)

## 背景

本地代理 `http://127.0.0.1:9567` 阻塞了 git push (git 协议走 22 端口，HTTPS 443 端口被代理拒绝)。

解决方案：用 GitHub REST API (Contents API) 逐文件推送，不走 git 协议。

## Token

Classic PAT (recommended): `<YOUR_TOKEN_HERE>`
Fine-Grained PAT 有权限粒度限制，写操作容易 403，建议用 Classic PAT。

## 推送脚本 (Python)

```python
import base64, os, requests, time

TOKEN = "<YOUR_TOKEN_HERE>"
REPO = "lemon890918-png/lottory-3d-analysis"
headers = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"}

def get_sha(path):
    r = requests.get(f"https://api.github.com/repos/{REPO}/contents/{path}", headers=headers, timeout=15)
    if r.status_code == 200:
        return r.json().get("sha")
    return None

def push_file(path, content, msg):
    sha = get_sha(path)
    data = {"message": msg, "content": base64.b64encode(content).decode()}
    if sha:
        data["sha"] = sha
    r = requests.put(f"https://api.github.com/repos/{REPO}/contents/{path}", headers=headers, json=data, timeout=20)
    return r.status_code

base = "/Users/wenxin/work/lottory-3d-analysis"
skip_dirs = {'.git'}
skip_ext = {'.pkl', '.png', '.jpg', '.jpeg', '.gif', '.csv', '.txt', '.json'}

msg = "feat: sync all project files"

for root, dirs, fnames in os.walk(base):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for f in fnames:
        ext = os.path.splitext(f)[1].lower()
        if ext in skip_ext or f.endswith('.pkl'):
            continue
        fp = os.path.join(root, f)
        rel = os.path.relpath(fp, base)
        with open(fp, 'rb') as fh:
            content = fh.read()
        status = push_file(rel, content, msg)
        print(f"{'OK' if status in (200,201) else 'FAIL'} {status}: {rel}")
        time.sleep(0.3)
```

## 注意

- 大文件（>100MB）不适合 Contents API，如需备份用 GitHub LFS
- .pkl/.png/.csv/.txt 等二进制/大文件默认跳过
- 每次推送间隔 0.3s 避免 rate limit（60次/小时）
