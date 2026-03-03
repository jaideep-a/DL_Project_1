---
description: How to push changes for this repo - always use the final_version branch
---

# Git Push Rules for DL_Project_1

**IMPORTANT: All code changes must be pushed to the `final_version` branch. Never push directly to `main` or any other branch.**

## Steps

1. Stage your changes:
```bash
git add -A
```

2. Commit with a descriptive message:
```bash
git commit -m "your descriptive message"
```

3. Push to the `final_version` branch:
```bash
git push origin final_version
```

## Rules
- Always target branch: `final_version`
- Never push to `main`
- Always check which branch you're on before pushing (`git branch`)
- If not on `final_version`, switch first: `git checkout final_version`
